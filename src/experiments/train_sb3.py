import argparse
import yaml
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# --- Import SB3 and the new wrapper ---
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_checker import check_env

from src.utils.gym_wrapper import GridWorldGymWrapper # Our new wrapper

def main():
    parser = argparse.ArgumentParser(description='Train Stable-Baselines3 on GridWorld')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--algo', type=str, default='DQN',
                       help='SB3 algorithm (PPO, A2C, DQN)')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Total training timesteps')
    parser.add_argument('--save-path', type=str, default='models/sb3_model',
                       help='Path to save the model')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Configuration:")
    print(yaml.dump(config, default_flow_style=False))

    # Set seed
    if config['env']['seed'] is not None:
        set_random_seed(config['env']['seed'])

    # --- Create the environment ---
    # Pass the env config to the wrapper
    env_kwargs = {'env_config': config['env']}
    # make_vec_env automatically vectorizes the environment (SB3 standard)
    env = make_vec_env(GridWorldGymWrapper, n_envs=1, env_kwargs=env_kwargs)

    # (Optional) Check if the environment is compliant
    # check_env(env.envs[0]) 

    # --- Select and initialize the SB3 algorithm ---
    algo_map = {
        'PPO': PPO,
        'A2C': A2C,
        'DQN': DQN
    }
    if args.algo not in algo_map:
        raise ValueError(f"Unknown algorithm: {args.algo}. Choose from {list(algo_map.keys())}")
        
    Algorithm = algo_map[args.algo]
    
    # Define model parameters
    policy = "MlpPolicy" # Multi-Layer Perceptron policy, good for vector inputs
    log_dir = config['logging']['log_dir']
    
    # We can re-use some of your config parameters for SB3's DQN
    if args.algo == 'DQN':
        model = Algorithm(
            policy, 
            env, 
            verbose=1, 
            buffer_size=config['train']['replay_size'],
            learning_rate=config['train']['lr'],
            gamma=config['train']['gamma'],
            exploration_initial_eps=config['train']['epsilon_start'],
            exploration_final_eps=config['train']['epsilon_end'],
            tensorboard_log=log_dir
        )
    else: # PPO or A2C
        model = Algorithm(
            policy, 
            env, 
            verbose=1,
            gamma=config['train']['gamma'],
            learning_rate=config['train']['lr'],
            tensorboard_log=log_dir
        )
    
    # --- Train the agent ---
    print(f"\nTraining {args.algo}...")
    model.learn(total_timesteps=args.timesteps, tb_log_name=f"{args.algo}_run")
    
    # --- Save the agent ---
    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
    model_save_file = f"{args.save_path}_{args.algo}"
    model.save(model_save_file)
    print(f"Model saved to {model_save_file}.zip")
    
    # --- Evaluate the trained agent ---
    print("\nEvaluating trained agent...")
    # Create a separate, non-vectorized eval env
    eval_env = GridWorldGymWrapper(env_config=config['env'])
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Evaluation: Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

    # --- Visualize the agent ---
    print("\nRunning one episode with rendering...")
    vis_env = GridWorldGymWrapper(env_config=config['env'], render_mode='human')
    obs, info = vis_env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = vis_env.step(action)
        if terminated or truncated:
            done = True
    vis_env.close()

if __name__ == '__main__':
    main()