import os
import time
from stable_baselines3 import PPO
import supersuit as ss
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

# Import the AEC environment constructor
from bus_marl_env import aec_env

# --- Configuration ---
MODEL_PATH = "models/ppo_sumo_bus_marl.zip" # Path to the trained model
NUM_EPISODES = 10
USE_GUI = True # Watch the evaluation in SUMO GUI

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        exit()

    print("Loading model...")
    model = PPO.load(MODEL_PATH)
    print("Model loaded.")

    print("Initializing evaluation environment...")
    # Create the AEC environment for evaluation
    eval_env_lambda = lambda: aec_env(use_gui=USE_GUI, render_mode='human' if USE_GUI else 'rgb_array')

    # Wrap it for SB3 using supersuit (similar to training)
    eval_vec_env = ss.pettingzoo_env_to_vec_env_v1(eval_env_lambda())
    # No need for VecMonitor usually during evaluation unless you want stats

    print("Environment Initialized for Evaluation.")

    total_rewards = {agent: 0.0 for agent in eval_vec_env.possible_agents}
    total_steps = 0
    episode_lengths = []
    episode_rewards = [] # Store total reward per episode

    for episode in range(NUM_EPISODES):
        print(f"\n--- Starting Episode {episode + 1}/{NUM_EPISODES} ---")
        obs, info = eval_vec_env.reset()
        terminated = {agent: False for agent in eval_vec_env.possible_agents}
        truncated = {agent: False for agent in eval_vec_env.possible_agents}
        episode_reward_sum = 0
        ep_steps = 0

        # Need to handle the VecEnv structure for observations/dones
        num_envs = eval_vec_env.num_envs # Should be 1 for evaluation usually
        current_obs = obs

        while True:
            # Get actions from the model
            # The vectorized env expects observations for all parallel envs (even if just 1)
            # The model outputs actions for all parallel envs
            actions, _states = model.predict(current_obs, deterministic=True)

            # Step the vectorized environment
            # It returns obs, rewards, dones, infos for *each* parallel environment
            obs, rewards, dones, infos = eval_vec_env.step(actions)

            # Since we usually eval with num_envs=1, access the first element
            reward_this_step = rewards[0]
            done_this_step = dones[0] # This 'done' is True if the *episode* in that env ends (terminated or truncated)

            episode_reward_sum += reward_this_step
            ep_steps += 1

            # Update current observation for the next iteration
            current_obs = obs

            if USE_GUI:
                time.sleep(0.05) # Slow down GUI for observation

            if done_this_step:
                print(f"Episode {episode + 1} finished after {ep_steps} steps.")
                print(f"Episode Reward: {episode_reward_sum:.2f}")
                episode_lengths.append(ep_steps)
                episode_rewards.append(episode_reward_sum)
                total_steps += ep_steps
                # Accumulate total rewards (less meaningful with VecEnv reward sum)
                # total_rewards += episode_reward_sum
                break # Move to the next episode

    eval_vec_env.close()
    print("\n--- Evaluation Summary ---")
    print(f"Ran {NUM_EPISODES} episodes.")
    avg_steps = sum(episode_lengths) / NUM_EPISODES if NUM_EPISODES > 0 else 0
    avg_reward = sum(episode_rewards) / NUM_EPISODES if NUM_EPISODES > 0 else 0
    print(f"Average Episode Length: {avg_steps:.2f} steps")
    print(f"Average Episode Reward: {avg_reward:.2f}")