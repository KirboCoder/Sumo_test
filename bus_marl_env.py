import gymnasium as gym
import numpy as np
import pandas as pd
import random
import sumolib
import traci
import os
import sys
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers, parallel_to_aec
from pettingzoo.utils.conversions import parallel_wrapper_fn
import xml.etree.ElementTree as ET
import os # Should already be there
import sys # Should already be there
import sumolib # Should already be there
import sumolib.net # Should already be there

# --- Environment Constants ---
# Adjust these paths if SUMO is not in your system PATH
SUMO_BINARY_GUI = "sumo-gui"
SUMO_BINARY_CMD = "sumo"
# Make sure this path is correct
SUMO_CONFIG_FILE = os.path.join("sumo_files", "city.sumocfg")

MAX_STEPS = 3600  # Max steps per episode (matches simulation end time)
NUM_BUSES = 2
PASSENGER_SPAWN_RATE = 0.1 # Probability of a passenger spawning at a random junction each step
MAX_PASSENGERS_ON_BUS = 30
JUNCTION_PREFIX = ":" # Default prefix for internal junctions in SUMO

# Action mapping: 0: Stay (continue route), 1-N: Choose Nth outgoing valid edge
# The number of actions depends on the maximum number of outgoing edges from any junction
# We'll dynamically determine this, but set a reasonable upper bound for the space definition.
MAX_ACTION_CHOICES = 5 # Stay + up to 4 outgoing edges

# Reward structure
REWARD_PICKUP = 10.0
REWARD_STEP_PENALTY = -0.1
REWARD_WAITING_PENALTY_FACTOR = -0.01 # Penalty per step per waiting passenger

class SumoBusMARLEnv(ParallelEnv):
    metadata = {
        "name": "SumoBusMARL_v0",
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
    }

    def __init__(self, render_mode='human', use_gui=True, config_file=SUMO_CONFIG_FILE, num_buses=NUM_BUSES):
        super().__init__()
        print(f"Initializing SumoBusMARLEnv with config: {config_file}") # Debug print
        self.render_mode = render_mode
        self.use_gui = use_gui and (render_mode == 'human')
        self.sumo_binary = SUMO_BINARY_GUI if self.use_gui else SUMO_BINARY_CMD
        self.config_file = config_file
        self.num_buses = num_buses

        # --- PettingZoo Setup ---
        self.possible_agents = [f"bus_{i}" for i in range(self.num_buses)]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.agents = [] # List of active agents

        # --- SUMO Network Info ---
        self.traci_conn = None
        # This call will now use the updated XML parsing method
        self._load_sumo_network()

        # --- Action Space ---
        self.max_outgoing_edges = self._get_max_outgoing_edges()
        self.num_actions = 1 + self.max_outgoing_edges # Stay + max choices
        self.action_spaces = {
            agent: gym.spaces.Discrete(self.num_actions)
            for agent in self.possible_agents
        }

        # --- Observation Space ---
        low = np.array([0, 0, 0, 0])
        high = np.array([len(self.edge_list), MAX_PASSENGERS_ON_BUS, MAX_STEPS, 100])
        self.observation_spaces = {
            agent: gym.spaces.Box(low=low, high=high, dtype=np.float32)
            for agent in self.possible_agents
        }

        # --- Internal State ---
        self.current_step = 0
        self.bus_data = {}
        self.passenger_data = {}
        # Network lists are initialized in _load_sumo_network


    def _load_sumo_network(self):
        """Loads the SUMO network by parsing the config file to find the net path."""
        net_file_path = None
        try:
            if not os.path.exists(self.config_file):
                raise FileNotFoundError(f"SUMO config file not found at '{self.config_file}'")

            # Parse the XML configuration file
            tree = ET.parse(self.config_file)
            root = tree.getroot()

            # Find the <input> section
            input_element = root.find('input')
            if input_element is None:
                raise ValueError(f"Could not find <input> section in '{self.config_file}'")

            # Find the <net-file> element within <input>
            net_file_element = input_element.find('net-file')
            if net_file_element is None:
                raise ValueError(f"Could not find <net-file> element within <input> in '{self.config_file}'")

            # Get the 'value' attribute which contains the path
            net_file_path_from_config = net_file_element.get('value')
            if net_file_path_from_config is None:
                raise ValueError(f"<net-file> element in '{self.config_file}' is missing the 'value' attribute")

            print(f"Found net file value from config: '{net_file_path_from_config}'")
            net_file_path = net_file_path_from_config # Assign to the variable we'll use

            # Check if the path is absolute or relative
            if not os.path.isabs(net_file_path):
                # If relative, join it with the directory of the config file
                config_dir = os.path.dirname(self.config_file)
                potential_path = os.path.join(config_dir, net_file_path)
                print(f"Path is relative. Checking adjusted path: '{potential_path}'")
                if os.path.exists(potential_path):
                    net_file_path = potential_path
                else:
                    # Keep the original relative path from config if the adjusted one doesn't exist
                    # SUMO might handle relative paths itself, but checking helps debugging
                    print(f"Adjusted path '{potential_path}' not found. Using original value '{net_file_path}' from config.")
                    # We still need to check if THIS path exists
                    if not os.path.exists(net_file_path):
                         # If the original relative path also doesn't exist relative to CWD
                         # then raise the error based on the adjusted path attempt.
                         raise FileNotFoundError(f"Network file specified in config not found. Tried absolute/relative path: '{potential_path}' and original value '{net_file_path}'")

            else:
                 print("Path is absolute.")
                 if not os.path.exists(net_file_path):
                      raise FileNotFoundError(f"Absolute network file path specified in config not found: '{net_file_path}'")


            print(f"Attempting to load network from: '{net_file_path}'")
            self.net = sumolib.net.readNet(net_file_path)
            print("Successfully loaded SUMO network.")

            # Initialize network-dependent attributes
            self.junction_list = [j.getID() for j in self.net.getNodes() if not j.getID().startswith(JUNCTION_PREFIX)]
            self.edge_list = [e.getID() for e in self.net.getEdges()]
            self.edge_to_int = {edge_id: i for i, edge_id in enumerate(self.edge_list)}
            self.int_to_edge = {i: edge_id for edge_id, i in self.edge_to_int.items()}
            print(f"Network loaded: {len(self.junction_list)} junctions, {len(self.edge_list)} edges.")


        except ET.ParseError as xml_err:
            print(f"Error parsing SUMO config file '{self.config_file}': {xml_err}")
            sys.exit(1)
        except FileNotFoundError as fnf_err:
             print(f"Error: {fnf_err}")
             print(f"Please ensure the <net-file value='...'/> in '{self.config_file}' points to the correct .net.xml file and the file exists.")
             sys.exit(1)
        except ValueError as val_err:
            print(f"Error reading config file structure '{self.config_file}': {val_err}")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error loading SUMO network (using net file: {net_file_path}): {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def _get_max_outgoing_edges(self):
        """Find the maximum number of outgoing edges from any non-internal junction."""
        max_edges = 0
        print("Calculating max outgoing edges...") # Debug print
        for junction in self.net.getNodes():
             junction_id = junction.getID()
             # Skip internal junctions if they exist (nodes starting with ':')
             if not junction_id.startswith(JUNCTION_PREFIX):
                # junction.getOutgoing() returns a list of Edge objects
                outgoing_edges = junction.getOutgoing()

                # Get the unique IDs of these outgoing edges.
                # An edge object directly represents the outgoing road.
                outgoing_edge_ids = {edge.getID() for edge in outgoing_edges} # Use set comprehension for uniqueness

                num_unique_outgoing = len(outgoing_edge_ids)
                # print(f"Junction {junction_id}: {num_unique_outgoing} outgoing edges -> {outgoing_edge_ids}") # Optional detailed debug print

                max_edges = max(max_edges, num_unique_outgoing)

        print(f"Max unique outgoing edges found: {max_edges}")
        # Ensure at least 1 action choice (for 'stay') even if a node has 0 outgoing edges
        # (though a valid network shouldn't have terminal nodes like that unless intended)
        # The actual action space size is 1 (stay) + max_edges
        return max_edges # Return the max count of choices *excluding* 'stay'

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _start_sumo(self):
        """Starts a SUMO simulation process and connects TraCI."""
        sumo_cmd = [
            self.sumo_binary,
            "-c", self.config_file,
            "--step-length", "1",    # Ensure step length matches environment logic
            # REMOVE THIS LINE: "--remote-port", str(sumolib.miscutils.getFreeSocketPort()),
            "--quit-on-end",
            # "--collision.action", "remove", # Optional: Handle collisions
            "--waiting-time-memory", str(MAX_STEPS), # Store waiting times
            # "--no-warnings", "true", # Optional: Suppress SUMO warnings if needed
            # Add other necessary SUMO options here
        ]
        # traci.start will automatically find a free port and add --remote-port
        traci.start(sumo_cmd)
        self.traci_conn = traci # Store connection object

    def _get_observation(self, agent_id):
        """Generates the observation for a single agent."""
        if agent_id not in self.bus_data: # Bus might not exist yet or was removed
             # Return a default observation (e.g., all zeros)
             return np.zeros(self.observation_spaces[agent_id].shape, dtype=np.float32)

        bus_info = self.bus_data[agent_id]
        current_edge = bus_info.get('edge', '')
        passengers_on_bus = len(bus_info.get('passengers', set()))

        # --- Nearby Passenger Info (Simplified) ---
        num_waiting_nearby = 0
        total_wait_time_nearby = 0
        nearby_junctions = set()
        if current_edge and current_edge in self.edge_to_int:
            try:
                edge_obj = self.net.getEdge(current_edge)
                nearby_junctions.add(edge_obj.getFromNode().getID())
                nearby_junctions.add(edge_obj.getToNode().getID())
            except KeyError: # Edge might not be in network (e.g., internal)
                 pass

        # Include next junction if known
        next_junc = bus_info.get('next_junction')
        if next_junc:
            nearby_junctions.add(next_junc)

        # Consider junctions reachable from nearby junctions (1 hop) - adjust radius as needed
        # extended_nearby = set(nearby_junctions)
        # for junc_id in nearby_junctions:
        #      if not junc_id.startswith(JUNCTION_PREFIX):
        #           try:
        #                junc = self.net.getNode(junc_id)
        #                for edge in junc.getOutgoing():
        #                     extended_nearby.add(edge.getToNode().getID())
        #           except KeyError: pass # Junction might not exist

        # Count waiting passengers at these junctions
        waiting_pass_count = 0
        for p_id, p_data in self.passenger_data.items():
            if p_data['status'] == 'waiting':
                # Check if passenger origin is one of the nearby junctions
                if p_data['origin'] in nearby_junctions:
                     wait_time = self.current_step - p_data['start_time']
                     total_wait_time_nearby += wait_time
                     waiting_pass_count += 1

        avg_wait_time_nearby = (total_wait_time_nearby / waiting_pass_count) if waiting_pass_count > 0 else 0

        # --- Assemble Observation Vector ---
        # Ensure edge is mapped to an integer
        edge_numeric = self.edge_to_int.get(current_edge, -1) # Use -1 for unknown edges

        obs = np.array([
            edge_numeric,
            passengers_on_bus,
            avg_wait_time_nearby,
            waiting_pass_count
        ], dtype=np.float32)

        # Ensure observation fits the defined space bounds (clipping might be necessary depending on exact values)
        # obs = np.clip(obs, self.observation_spaces[agent_id].low, self.observation_spaces[agent_id].high)

        return obs


    def _apply_action(self, agent_id, action):
        """Translates agent action into SUMO commands."""
        if agent_id not in traci.vehicle.getIDList():
            # print(f"Warning: Agent {agent_id} not found in simulation.")
            return # Agent might have finished or crashed

        current_edge = traci.vehicle.getRoadID(agent_id)
        if not current_edge or current_edge.startswith(":"): # In junction or internal edge
            # print(f"Agent {agent_id} is at junction {current_edge}, cannot change target now.")
            # Let SUMO handle routing or just continue
            try:
                 # Ensure it continues if it has a route
                 if not traci.vehicle.getRoute(agent_id):
                      # Maybe assign a default next edge if possible? Complex.
                      pass # For now, do nothing if in junction without route
            except traci.TraCIException: pass # Vehicle might be gone
            return

        # Action 0: Stay/Continue default route
        if action == 0:
            # Ensure the bus continues its current route or stops if needed
            # If stopped, maybe start moving again? Depends on desired 'stay' behavior.
            # For now, just let SUMO manage the current route.
            # If you want 'stay' to mean 'stop at next valid stop':
            # traci.vehicle.setStop(agent_id, current_edge, ...)
            pass # Let SUMO handle default routing for now
            return

        # Action > 0: Choose an outgoing edge from the *next* junction
        try:
            route = traci.vehicle.getRoute(agent_id)
            if not route: return # No route, cannot determine next junction easily

            current_route_index = traci.vehicle.getRouteIndex(agent_id)

            # Find the next *junction* based on the route
            next_junction_id = None
            # Look ahead in the route for the next edge's target junction
            if current_route_index < len(route) -1:
                 next_edge_in_route = route[current_route_index + 1]
                 try:
                      next_junction_id = self.net.getEdge(next_edge_in_route).getToNode().getID()
                 except KeyError: pass # Edge not found in network data
            elif current_route_index == len(route) -1: # At the last edge
                 try:
                      next_junction_id = self.net.getEdge(current_edge).getToNode().getID()
                 except KeyError: pass # Edge not found

            if not next_junction_id or next_junction_id.startswith(JUNCTION_PREFIX):
                # print(f"Agent {agent_id}: Cannot determine valid next junction.")
                return # Cannot determine next junction or it's internal

            # Get valid outgoing edges from this next junction
            junction = self.net.getNode(next_junction_id)
            outgoing_links = junction.getOutgoing()
            valid_target_edges = []
            for link in outgoing_links:
                 via_lane = link.getViaLane()
                 if via_lane:
                      edge = via_lane.getEdge()
                      # Add filters here if needed (e.g., disallow u-turns, check road type)
                      # Basic filter: Don't go immediately back onto the current edge
                      if edge.getID() != current_edge:
                           valid_target_edges.append(edge.getID())

            valid_target_edges = sorted(list(set(valid_target_edges))) # Unique and sorted for consistency

            if not valid_target_edges:
                # print(f"Agent {agent_id}: No valid outgoing edges from {next_junction_id}.")
                return # No valid choices

            # Map action index (1 to N) to the chosen edge
            chosen_edge_index = action - 1
            if 0 <= chosen_edge_index < len(valid_target_edges):
                target_edge = valid_target_edges[chosen_edge_index]
                # Command SUMO to change the vehicle's target edge
                # Using changeTarget forces rerouting from the *current* edge towards the target
                # Using setRoute might be better if you want a specific path
                # print(f"Agent {agent_id} action {action}: Changing target to {target_edge}")
                traci.vehicle.changeTarget(agent_id, target_edge)
                # Optional: Update internal state about intended next junction
                self.bus_data[agent_id]['next_junction'] = self.net.getEdge(target_edge).getToNode().getID()

            else:
                # Invalid action index for the available choices, treat as 'stay'
                # print(f"Agent {agent_id}: Action {action} invalid for {len(valid_target_edges)} choices. Staying.")
                pass

        except traci.TraCIException as e:
            # print(f"TraCI Error applying action for {agent_id}: {e}")
            # This often happens if the vehicle leaves the simulation unexpectedly
            self._remove_agent(agent_id)
        except KeyError as e:
             # print(f"Network Error applying action for {agent_id}: {e}") # Edge/Node not found
             pass
        except Exception as e:
             print(f"Generic Error applying action for {agent_id}: {e}")


    def _update_passenger_state(self):
        """Handles passenger spawning, pickup, and arrival."""
        # 1. Spawn new passengers
        for junc_id in self.junction_list:
            if random.random() < PASSENGER_SPAWN_RATE:
                passenger_id = f"p_{self.current_step}_{junc_id}"
                # Choose a random destination different from origin
                possible_destinations = [j for j in self.junction_list if j != junc_id]
                if possible_destinations:
                     destination_id = random.choice(possible_destinations)
                     # Add person using TraCI - they need an initial route/stage
                     # Simplest: add person and hope a bus picks them up.
                     # A better way involves defining person stages (walk->bus->walk)
                     try:
                          # Find edges connected to the junctions for stage definition
                          origin_edge = random.choice(self.net.getNode(junc_id).getIncoming()).getID()
                          dest_edge = random.choice(self.net.getNode(destination_id).getOutgoing()).getID()

                          traci.person.add(passenger_id, origin_edge, pos=random.uniform(5, 15)) # Add near start of edge
                          # Add a basic trip stage - this is crucial for pickup logic
                          traci.person.appendDrivingStage(passenger_id, dest_edge, lines="BUS") # Target edge, specify bus lines

                          self.passenger_data[passenger_id] = {
                              'start_time': self.current_step,
                              'origin': junc_id, # Store junction ID
                              'destination': destination_id,
                              'status': 'waiting',
                              'bus': None
                          }
                     except (traci.TraCIException, IndexError, KeyError) as e:
                          # print(f"Error adding passenger {passenger_id}: {e}")
                          pass # Ignore if adding fails

        # 2. Check for pickups and update bus passenger lists
        picked_up_counts = {agent_id: 0 for agent_id in self.agents}
        boarded_this_step = set()
        try:
            for bus_id in self.bus_data.keys(): # Iterate over known buses
                if bus_id in traci.vehicle.getIDList(): # Check if bus still exists
                    current_passengers_in_sim = set(traci.vehicle.getPersonIDList(bus_id))
                    newly_boarded = current_passengers_in_sim - self.bus_data[bus_id]['passengers']

                    for p_id in newly_boarded:
                        if p_id in self.passenger_data and self.passenger_data[p_id]['status'] == 'waiting':
                            self.passenger_data[p_id]['status'] = 'riding'
                            self.passenger_data[p_id]['bus'] = bus_id
                            picked_up_counts[bus_id] += 1
                            boarded_this_step.add(p_id)
                            # print(f"Passenger {p_id} boarded {bus_id}")

                    # Update the bus's passenger set
                    self.bus_data[bus_id]['passengers'] = current_passengers_in_sim

        except traci.TraCIException as e:
            # print(f"TraCI error during pickup check: {e}")
            pass # Handle potential errors if vehicles disappear

        # 3. Check for arrived passengers (those who completed their trip)
        arrived_person_ids = traci.simulation.getArrivedPersonIDs()
        for p_id in arrived_person_ids:
            if p_id in self.passenger_data:
                self.passenger_data[p_id]['status'] = 'arrived'
                # print(f"Passenger {p_id} arrived.")
                # Optionally remove from passenger_data if no longer needed
                # del self.passenger_data[p_id]

        return picked_up_counts


    def _calculate_rewards(self, picked_up_counts):
        """Calculates rewards for each agent."""
        rewards = {agent_id: 0.0 for agent_id in self.agents}

        # Base step penalty
        for agent_id in self.agents:
             if agent_id in self.bus_data: # Only penalize active buses
                  rewards[agent_id] += REWARD_STEP_PENALTY

        # Pickup rewards
        for agent_id, count in picked_up_counts.items():
            if agent_id in rewards:
                rewards[agent_id] += count * REWARD_PICKUP

        # Waiting time penalty (applied globally or per bus based on assignment?)
        # Global penalty applied to all agents:
        total_waiting_time = 0
        waiting_count = 0
        for p_data in self.passenger_data.values():
            if p_data['status'] == 'waiting':
                total_waiting_time += (self.current_step - p_data['start_time'])
                waiting_count += 1

        global_penalty = total_waiting_time * REWARD_WAITING_PENALTY_FACTOR
        for agent_id in rewards:
             rewards[agent_id] += global_penalty # Distribute penalty

        return rewards

    def reset(self, seed=None, options=None):
        """Resets the environment to a starting state."""
        print("Resetting environment...") # Debug print
        if self.traci_conn:
            try:
                traci.close()
            except Exception as e:
                print(f"Ignoring error during TraCI close on reset: {e}")
            finally:
                self.traci_conn = None

        # Set seed if provided
        if seed is not None:
             random.seed(seed)
             np.random.seed(seed)

        try:
            self._start_sumo()
        except Exception as e:
             print(f"Failed to start SUMO during reset: {e}")
             if self.traci_conn:
                  try: traci.close()
                  except: pass
                  self.traci_conn = None
             raise RuntimeError(f"Could not start SUMO simulation in reset: {e}") from e

        self.current_step = 0
        self.agents = self.possible_agents[:]
        self.passenger_data = {}
        self.bus_data = {agent_id: {'edge': None, 'passengers': set(), 'next_junction': None} for agent_id in self.agents}

        # --- Add initial buses if not defined in rou.xml ---
        # Make sure your city.rou.xml defines the vehicles bus_0, bus_1, ...
        # Or uncomment and adapt this section if you need dynamic adding:
        # try:
        #     existing_buses_in_sim = traci.vehicle.getIDList()
        #     print(f"Buses defined in rou.xml (or already added): {existing_buses_in_sim}")
        #     for i, agent_id in enumerate(self.possible_agents):
        #          if agent_id not in existing_buses_in_sim:
        #               # Choose a valid starting edge from your network
        #               # start_edge = random.choice(self.edge_list) # Example: random start
        #               start_edge = "westTop___intersectionNW" # Example: specific start edge
        #               if start_edge not in self.edge_to_int:
        #                    print(f"Warning: Chosen start edge '{start_edge}' not in network edge list. Check edge IDs.")
        #                    # Fallback or error handling needed here
        #                    valid_edges = [e for e in self.edge_list if not e.startswith(':')]
        #                    if not valid_edges: raise ValueError("No valid non-internal edges found in the network to start a bus.")
        #                    start_edge = random.choice(valid_edges)
        #                    print(f"Using fallback start edge: {start_edge}")

        #               print(f"Attempting to dynamically add {agent_id} departing at step {i*2} on edge {start_edge}")
        #               # Use add 'full' to specify type, route, etc.
        #               traci.vehicle.add(vehID=agent_id, routeID="", typeID="BUS", depart=str(i*2), departLane="best", departPos="base", departSpeed="max")
        #               traci.vehicle.changeTarget(agent_id, start_edge) # Give it an initial target edge
        #               print(f"Dynamically added {agent_id}")
        # except traci.TraCIException as e:
        #      print(f"TraCI Error during dynamic bus adding: {e}")
        #      # Handle failure - maybe raise error?
        #      raise RuntimeError(f"Failed to add bus {agent_id} dynamically: {e}") from e
        # except Exception as e:
        #      print(f"Unexpected error during dynamic bus adding: {e}")
        #      raise


                # Initial simulation step
        try:
            print("Performing initial simulation step...")
            traci.simulationStep()
            self.current_step += 1
            print("Initial step complete.")
        except traci.TraCIException as e:
            print(f"TraCI Error during initial simulation step: {e}. SUMO likely closed.")
            self.close()
            raise RuntimeError(f"SUMO closed unexpectedly during initial step: {e}") from e

        # Verify all agents exist and update initial data
        try:
            current_sim_vehicles = set(traci.vehicle.getIDList())
            print(f"Vehicles present after initial step: {current_sim_vehicles}")
            missing_agents = []
            for agent_id in self.possible_agents:
                if agent_id in current_sim_vehicles:
                    try:
                        self.bus_data[agent_id]['edge'] = traci.vehicle.getRoadID(agent_id)
                        print(f"Agent {agent_id} confirmed at edge: {self.bus_data[agent_id]['edge']}")
                    except traci.TraCIException as e:
                        print(f"TraCI Error getting initial state for {agent_id} (might have crashed immediately): {e}")
                        missing_agents.append(agent_id)
                else:
                    print(f"Error: Agent {agent_id} not found in simulation after initial step!")
                    missing_agents.append(agent_id)

            if missing_agents:
                self.close()
                raise RuntimeError(f"Environment reset failed: Agents {missing_agents} did not spawn correctly or disappeared immediately. Check route file and SUMO logs.")

        except traci.TraCIException as e:
             print(f"TraCI Error verifying agents after initial step: {e}")
             self.close()
             raise RuntimeError(f"SUMO connection lost while verifying initial agent state: {e}") from e

         # Get initial observations for all possible agents
        observations = {agent_id: self._get_observation(agent_id) for agent_id in self.possible_agents}
        # REINSTATE INFOS DICTIONARY
        infos = {agent_id: {} for agent_id in self.possible_agents}

        print(f"Reset complete. Active agents: {self.agents}")
        # Return BOTH observations and infos as per PettingZoo standard
        return observations, infos


    def step(self, actions):
        """Advances the environment by one step."""
        if not self.agents or self.current_step >= MAX_STEPS:
            # Episode ended, return empty values
            self.agents = [] # Ensure agents list is empty
            return {}, {}, {}, {}, {}

        # 1. Apply actions for each agent
        for agent_id in self.agents:
            if agent_id in actions:
                self._apply_action(agent_id, actions[agent_id])

        # 2. Advance SUMO simulation
        try:
            traci.simulationStep()
            self.current_step += 1
        except traci.TraCIException as e:
            print(f"TraCI Error during simulation step: {e}. Ending episode.")
            traci.close()
            self.traci_conn = None
            # Terminate all agents
            observations = {agent_id: self._get_observation(agent_id) for agent_id in self.agents}
            rewards = {agent_id: 0 for agent_id in self.agents} # Or some penalty
            terminations = {agent_id: True for agent_id in self.agents}
            truncations = {agent_id: False for agent_id in self.agents} # Not truncated, but terminated due to error
            infos = {agent_id: {'error': str(e)} for agent_id in self.agents}
            self.agents = []
            return observations, rewards, terminations, truncations, infos


        # 3. Update environment state (passengers, bus positions)
        picked_up_counts = self._update_passenger_state()

        # Update bus positions and check for removals
        current_sim_buses = set(traci.vehicle.getIDList())
        agents_to_remove = []
        for agent_id in self.agents:
            if agent_id in current_sim_buses:
                 try:
                      self.bus_data[agent_id]['edge'] = traci.vehicle.getRoadID(agent_id)
                      # Update next junction based on current route? More complex.
                 except traci.TraCIException:
                      # Bus likely removed (e.g., collision, arrived)
                      agents_to_remove.append(agent_id)
                 except KeyError: # Bus might be on an edge not in our initial list (internal?)
                      self.bus_data[agent_id]['edge'] = None # Mark as unknown
            else: # Bus no longer in simulation
                 agents_to_remove.append(agent_id)

        for agent_id in agents_to_remove:
             self._remove_agent(agent_id)

        # 4. Calculate rewards
        rewards = self._calculate_rewards(picked_up_counts)

        # 5. Check for terminations and truncations
        terminations = {agent_id: False for agent_id in self.agents}
        truncations = {agent_id: False for agent_id in self.agents}

        if self.current_step >= MAX_STEPS:
            truncations = {agent_id: True for agent_id in self.agents}
            # print("Max steps reached, truncating episode.")
            self.agents = [] # End episode for all

        # Check if individual agents finished (e.g., reached a destination - needs logic)
        # for agent_id in agents_to_remove:
        #      if agent_id not in terminations: # If not already removed for other reasons
        #           terminations[agent_id] = True # Mark as terminated

        # If all agents are gone (removed or finished)
        if not self.agents:
             # Ensure all remaining possible agents are marked done if they weren't active
             for agent_id in self.possible_agents:
                  if agent_id not in terminations: terminations[agent_id] = True
                  if agent_id not in truncations: truncations[agent_id] = False # Terminated, not truncated
             # print("All agents finished or removed.")


        # 6. Get next observations
        observations = {agent_id: self._get_observation(agent_id) for agent_id in self.agents}

        # 7. Prepare info dictionary
        infos = {agent_id: {} for agent_id in self.agents}
        # Add optional debug info
        # for agent_id in self.agents:
        #      infos[agent_id]['passengers'] = len(self.bus_data[agent_id]['passengers'])
        #      infos[agent_id]['current_edge'] = self.bus_data[agent_id]['edge']


        # If the simulation ended gracefully via SUMO's end time
        if traci.simulation.getMinExpectedNumber() <= 0 and not any(truncations.values()):
             print("SUMO simulation ended gracefully.")
             truncations = {agent_id: True for agent_id in self.agents} # Truncate if SUMO ends before MAX_STEPS
             self.agents = []

        # Filter outputs to only include active agents for this step
        active_rewards = {a: rewards.get(a, 0) for a in self.agents}
        active_terminations = {a: terminations.get(a, False) for a in self.agents}
        active_truncations = {a: truncations.get(a, False) for a in self.agents}

        return observations, active_rewards, active_terminations, active_truncations, infos


    def render(self):
        """Rendering is handled by SUMO-GUI if enabled."""
        if self.render_mode == "human":
            # The GUI runs in a separate process. No direct rendering command here.
            # You could potentially add Pygame or Matplotlib visualizations
            # based on the environment state if needed.
            pass
        elif self.render_mode == "rgb_array":
             # Requires screenshot capability or alternative visualization library
             print("RGB array rendering not implemented. Use SUMO-GUI ('human' mode).")
             return None

    # Make sure the close method correctly uses self.traci_conn
    def close(self):
        """Closes the TraCI connection."""
        if self.traci_conn: # Check if it was ever assigned (not None)
            try:
                traci.close()
                # print("TraCI connection closed.")
            except Exception as e:
                print(f"Error closing TraCI: {e}")
            finally:
                 self.traci_conn = None # Set to None after closing or if error occurs

    # __del__ remains the same, but will now work because self.traci_conn exists
    def __del__(self):
        """Ensure TraCI connection is closed when the object is deleted."""
        self.close()


# --- Wrappers for SB3 Compatibility ---
# Using AEC environment for SB3 PPO/MaskablePPO is often recommended
def aec_env(**kwargs):
    env = SumoBusMARLEnv(**kwargs)
    # Add necessary AEC wrappers if needed (e.g., order enforcement)
    # env = wrappers.OrderEnforcingWrapper(env) # Usually needed for AEC
    # Convert to AEC
    aec_env = parallel_to_aec(env)
    # AEC environments often need assert_order_wrapper
    aec_env = wrappers.AssertOutOfBoundsWrapper(aec_env)
    aec_env = wrappers.OrderEnforcingWrapper(aec_env)
    return aec_env

# Wrapper for using ParallelEnv directly with SB3 (requires specific SB3 setup)
# from supersuit.multiagent_wrappers import pettingzoo_env_to_vec_env_v1
# def parallel_env_wrapper(**kwargs):
#     env = SumoBusMARLEnv(**kwargs)
#     vec_env = pettingzoo_env_to_vec_env_v1(env)
#     # Add SB3 VecEnv wrappers like VecMonitor if needed
#     # from stable_baselines3.common.vec_env import VecMonitor
#     # vec_env = VecMonitor(vec_env)
#     return vec_env