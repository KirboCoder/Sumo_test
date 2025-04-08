import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
import supersuit as ss # Keep this import

# Import the PARALLEL environment class directly
from bus_marl_env import SumoBusMARLEnv

# --- Configuration ---
USE_GUI = False
TOTAL_TIMESTEPS = 200_000 # Adjust as needed
MODEL_SAVE_PATH = "models/ppo_sumo_bus_marl"
LOG_PATH = "logs/"
NUM_CPUS = 1 # Keep this at 1 for now

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

if __name__ == "__main__":
    print("Initializing environment...")

    parallel_env = SumoBusMARLEnv(use_gui=USE_GUI, render_mode='rgb_array')

    print("Wrapping ParallelEnv to VecEnv using SuperSuit...")
    # Try the _v1 wrapper name directly under ss
    vec_env = ss.pettingzoo_env_to_vec_env_v1(parallel_env)

    # Wrap with VecMonitor for logging
    vec_env = VecMonitor(vec_env, LOG_PATH)
    print("VecEnv created and wrapped with Monitor.")

    print(f"Observation Space: {vec_env.observation_space}")
    print(f"Action Space: {vec_env.action_space}")

    # --- Model Setup ---
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4, # Often needs tuning
        verbose=1,
        tensorboard_log=LOG_PATH
    )

    print("Starting training...")
    start_time = time.time()
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            log_interval=1, # Log every N updates
            tb_log_name="PPO_BusMARL"
        )
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure environment is closed properly
        vec_env.close()

    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")

    # --- Save Final Model ---
    print(f"Saving final model to {MODEL_SAVE_PATH}")
    model.save(MODEL_SAVE_PATH)

    print("Training complete.")
    print(f"To view logs: tensorboard --logdir {LOG_PATH}")