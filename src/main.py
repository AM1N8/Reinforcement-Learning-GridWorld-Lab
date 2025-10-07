import argparse
import numpy as np
import matplotlib.pyplot as plt
from model import GridWorldModel
from agent import Agent
from environment import GridWorldEnv
from visualizer import GridWorldVisualizer
from algorithms import ValueIteration, PolicyIteration, MonteCarlo, QLearning, SarsaLambda
from dqn import DQN
import os
import json
from datetime import datetime

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='GridWorld Reinforcement Learning')
    
    # Grid configuration
    parser.add_argument('--rows', type=int, default=5, help='Number of rows')
    parser.add_argument('--cols', type=int, default=5, help='Number of columns')
    
    # Start states
    parser.add_argument('--starts', nargs='+', type=str, default=['0,0'],
                       help='Start states as row,col')
    
    # Goal states
    parser.add_argument('--goals', nargs='+', type=str, default=None,
                       help='Goal states as row,col')
    
    # Obstacles
    parser.add_argument('--obstacles', nargs='+', type=str, default=None,
                       help='Obstacle positions as row,col')
    
    # Rewards
    parser.add_argument('--goal-reward', type=float, default=10.0,
                       help='Reward for reaching goal')
    parser.add_argument('--step-penalty', type=float, default=-1.0,
                       help='Penalty per step')
    parser.add_argument('--obstacle-penalty', type=float, default=-5.0,
                       help='Penalty for hitting obstacle')
    
    # Algorithm selection
    parser.add_argument('--algorithm', type=str, 
                       choices=['random', 'vi', 'pi', 'mc', 'ql', 'sl', 'dql', 'compare'],
                       default='ql', help='Algorithm to run')
    
    # Algorithm hyperparameters
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Exploration rate')
    parser.add_argument('--alpha', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--episodes', type=int, default=500, help='Training episodes')
    parser.add_argument('--max-steps', type=int, default=100, help='Max steps per episode')
    parser.add_argument('--lambda-val', type=float, default=0.9, help='Lambda for SARSA(λ)')
    
    # DQN specific
    parser.add_argument('--dqn-hidden', nargs='+', type=int, default=[128, 128],
                       help='Hidden layer sizes for DQN')
    parser.add_argument('--dqn-buffer-size', type=int, default=10000,
                       help='Replay buffer size')
    parser.add_argument('--dqn-batch-size', type=int, default=64,
                       help='Batch size for DQN')
    parser.add_argument('--dqn-target-update', type=int, default=10,
                       help='Target network update frequency')
    parser.add_argument('--epsilon-decay', type=float, default=0.995,
                       help='Epsilon decay rate')
    
    # Visualization
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--save-dir', type=str, default='../visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--animate', action='store_true',
                       help='Run episode with animation')
    parser.add_argument('--save-animation', action='store_true', 
                       help='Save animation as GIF')
    parser.add_argument('--animate-learning', action='store_true',
                       help='Animate learning process episode-by-episode')
    
    # Model saving/loading
    parser.add_argument('--save-model', action='store_true',
                       help='Save trained model')
    parser.add_argument('--load-model', type=str, default=None,
                       help='Load model from path')
    parser.add_argument('--save-logs', action='store_true',
                       help='Save training logs')
    parser.add_argument('--load-logs', type=str, default=None,
                       help='Load and visualize training logs')
    
    # Grid size experiments
    parser.add_argument('--grid-size-experiment', action='store_true',
                       help='Run grid size scaling experiment')
    parser.add_argument('--grid-sizes', nargs='+', type=str, default=['3x3', '5x5', '7x7', '10x10'],
                       help='Grid sizes for experiment')

    return parser.parse_args()


def parse_positions(pos_strings):
    """Parse position strings like "1,2" into tuples"""
    if pos_strings is None:
        return None
    positions = []
    for pos_str in pos_strings:
        row, col = map(int, pos_str.split(','))
        positions.append((row, col))
    return positions


def create_model(args):
    """Create GridWorld model from arguments"""
    starts = parse_positions(args.starts)
    goals = parse_positions(args.goals)
    obstacles = parse_positions(args.obstacles)
    
    model = GridWorldModel(
        rows=args.rows,
        cols=args.cols,
        start_states=starts,
        goal_states=goals,
        obstacles=obstacles,
        goal_reward=args.goal_reward,
        step_penalty=args.step_penalty,
        obstacle_penalty=args.obstacle_penalty
    )
    
    return model


def run_episode(env, agent, max_steps=50, verbose=True):
    """Run a single episode"""
    state, info = env.reset(seed=42)
    agent.reset()
    
    if verbose:
        print("Starting episode...")
        env.render()
    
    for step in range(max_steps):
        action = agent.select_action(state, training=False)
        
        if verbose:
            action_name = env.model.action_names[action]
            print(f"\nStep {step + 1}: Action = {action_name}")
        
        next_state, reward, terminated, truncated, info = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state
        
        if verbose:
            print(f"Position: {state}, Reward: {reward:.1f}, Total: {agent.total_reward:.1f}")
        
        if terminated:
            if verbose:
                print(f"\nGoal reached in {agent.steps} steps!")
                print(f"Total reward: {agent.total_reward:.1f}")
            break
    
    return env.get_trajectory(), agent.total_reward, agent.steps


def run_animated_episode(args, model, algorithm, title):
    """Run a single episode with animation"""
    print("\n" + "=" * 60)
    print(f"Running Animated Episode for {title}")
    print("=" * 60)
    
    env = GridWorldEnv(model=model, render_mode=None) 
    agent = Agent(num_actions=model.num_actions, policy="learned", algorithm=algorithm)
    
    state, info = env.reset(seed=42)
    agent.reset()
    trajectory = [state.copy()]
    
    for step in range(args.max_steps):
        action = agent.select_action(state, training=False)
        next_state, reward, terminated, truncated, info = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state
        trajectory.append(state.copy())
        
        if terminated or truncated:
            break
            
    print(f"Episode completed. Steps: {agent.steps}, Total Reward: {agent.total_reward:.1f}")

    save_path = None
    if args.save_animation:
        alg_name = title.split(' ')[0].lower()
        file_name = f"{alg_name}_trajectory.gif"
        save_path = os.path.join(args.save_dir, file_name)
    
    viz = GridWorldVisualizer(rows=args.rows, cols=args.cols,
                             goal_states=model.goal_states,
                             obstacles=model.obstacles,
                             alpha=getattr(algorithm, 'alpha', None), 
                             gamma=args.gamma,
                             epsilon=getattr(algorithm, 'epsilon', None))
    
    viz.animate_episode(trajectory, interval_ms=500, title=title, save_path=save_path) 
    env.close()


def collect_episode_data(env, algorithm, num_episodes, max_steps):
    """Collect trajectory data for each episode during training"""
    episode_trajectories = []
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        trajectory = [state.copy()]
        total_reward = 0
        
        for step in range(max_steps):
            action = algorithm.get_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            trajectory.append(next_state.copy())
            total_reward += reward
            
            # Update algorithm (simplified - actual update depends on algorithm)
            if hasattr(algorithm, 'replay_buffer'):
                algorithm.replay_buffer.push(state, action, reward, next_state, float(done))
                algorithm.update()
            
            state = next_state
            
            if done:
                break
        
        episode_trajectories.append(trajectory)
        episode_rewards.append(total_reward)
        
        # Decay epsilon if applicable
        if hasattr(algorithm, 'epsilon'):
            algorithm.epsilon = max(algorithm.epsilon_end, 
                                   algorithm.epsilon * algorithm.epsilon_decay)
    
    return episode_trajectories, episode_rewards


def run_dql(args, model):
    """Run Deep Q-Learning"""
    print("\n" + "=" * 60)
    print("Deep Q-Learning (DQL)")
    print("=" * 60)
    
    env = GridWorldEnv(model=model)
    
    # Create DQN agent
    dqn = DQN(model=model, state_dim=2, hidden_dims=args.dqn_hidden,
              alpha=args.alpha, gamma=args.gamma,
              epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=args.epsilon_decay,
              buffer_size=args.dqn_buffer_size, batch_size=args.dqn_batch_size,
              target_update_freq=args.dqn_target_update)
    
    # Load model if specified
    if args.load_model:
        dqn.load_model(args.load_model)
        print(f"Loaded model from {args.load_model}")
    else:
        print("\nTraining DQN...")
        
        # Collect episode data if animation is needed
        if args.animate_learning:
            episode_trajectories, episode_rewards = collect_episode_data(
                env, dqn, args.episodes, args.max_steps)
        else:
            Q, episode_rewards = dqn.train(env, num_episodes=args.episodes,
                                          max_steps=args.max_steps,
                                          verbose_freq=50)
    
    # Save model if specified
    if args.save_model:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(args.save_dir, f"dqn_model_{timestamp}.pth")
        dqn.save_model(model_path)
    
    # Save training logs
    if args.save_logs:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(args.save_dir, f"dqn_log_{timestamp}.json")
        dqn.save_training_log(log_path)
    
    # Test trained agent
    agent = Agent(num_actions=model.num_actions, policy="learned", algorithm=dqn)
    trajectory, reward, steps = run_episode(env, agent, max_steps=args.max_steps,
                                           verbose=args.verbose)
    
    print("\nDQN Results:")
    print(f"Steps: {steps}, Total Reward: {reward:.1f}")
    print(f"Exploration ratio: {dqn.get_exploration_ratio():.3f}")
    
    # Animate single episode
    if args.animate:
        run_animated_episode(args, model, dqn, "Deep Q-Learning")
    
    # Animate learning process
    if args.animate_learning and 'episode_trajectories' in locals():
        viz = GridWorldVisualizer(rows=args.rows, cols=args.cols,
                                 goal_states=model.goal_states,
                                 obstacles=model.obstacles,
                                 alpha=args.alpha, gamma=args.gamma)
        
        save_path = None
        if args.save_animation:
            save_path = os.path.join(args.save_dir, "dql_learning_process.gif")
        
        viz.animate_learning_process(episode_trajectories, episode_rewards,
                                     interval_ms=800, title="DQL Learning Process",
                                     save_path=save_path)
    
    # Visualizations
    if not args.no_viz:
        viz = GridWorldVisualizer(rows=args.rows, cols=args.cols,
                                 goal_states=model.goal_states,
                                 obstacles=model.obstacles,
                                 alpha=args.alpha, gamma=args.gamma,
                                 epsilon=dqn.epsilon)
        
        # Q-values
        fig_q = viz.plot_q_values(dqn.Q, model, "DQL - Q-Values")
        viz.save_figure(fig_q, "dql_q_values.png", args.save_dir)
        
        # Trajectory
        fig_traj, _ = viz.plot_trajectory(trajectory, "DQL - Trajectory")
        viz.save_figure(fig_traj, "dql_trajectory.png", args.save_dir)
        
        # Learning curve
        fig_lc = viz.plot_learning_curve(dqn.episode_rewards, "DQL - Learning Curve")
        viz.save_figure(fig_lc, "dql_learning_curve.png", args.save_dir)
        
        # Epsilon decay
        fig_eps = viz.plot_epsilon_decay(dqn.epsilon_history, "DQL - Epsilon Decay")
        viz.save_figure(fig_eps, "dql_epsilon_decay.png", args.save_dir)
        
        # Training summary
        metrics = {
            'episode_rewards': dqn.episode_rewards,
            'episode_steps': dqn.episode_steps,
            'epsilon_history': dqn.epsilon_history,
            'loss_history': dqn.loss_history,
            'exploration_count': dqn.exploration_count,
            'exploitation_count': dqn.exploitation_count
        }
        fig_summary = viz.plot_training_summary(metrics, "DQL - Training Summary")
        viz.save_figure(fig_summary, "dql_training_summary.png", args.save_dir)
        
        plt.show()
    
    env.close()


def run_random(args, model):
    """Run random agent"""
    print("\n" + "=" * 60)
    print("Testing Random Agent")
    print("=" * 60)
    
    env = GridWorldEnv(model=model, render_mode="human" if args.verbose else None)
    agent = Agent(num_actions=model.num_actions, policy="random")
    trajectory, reward, steps = run_episode(env, agent, max_steps=args.max_steps, 
                                           verbose=args.verbose)
    env.close()
    
    print("\nRandom Agent Results:")
    print(f"Steps: {steps}, Total Reward: {reward:.1f}")
    
    if not args.no_viz:
        viz = GridWorldVisualizer(rows=args.rows, cols=args.cols, 
                                 goal_states=model.goal_states,
                                 obstacles=model.obstacles)
        fig, ax = viz.plot_trajectory(trajectory, "Random Agent")
        plt.show()


def run_value_iteration(args, model):
    """Run Value Iteration"""
    print("\n" + "=" * 60)
    print("Value Iteration")
    print("=" * 60)
    
    env = GridWorldEnv(model=model)
    
    print("\nTraining Value Iteration...")
    vi = ValueIteration(model, gamma=args.gamma)
    V, policy = vi.train(max_iterations=1000)
    
    agent = Agent(num_actions=model.num_actions, policy="learned", algorithm=vi)
    trajectory, reward, steps = run_episode(env, agent, max_steps=args.max_steps,
                                           verbose=args.verbose)
    
    print("\nValue Iteration Results:")
    print(f"Steps: {steps}, Total Reward: {reward:.1f}")
    
    if args.animate:
        run_animated_episode(args, model, vi, "Value Iteration") 
        
    if not args.no_viz:
        viz = GridWorldVisualizer(rows=args.rows, cols=args.cols,
                                 goal_states=model.goal_states,
                                 obstacles=model.obstacles,
                                 gamma=args.gamma)
                                 
        fig_v, _ = viz.plot_value_function(V, "Value Iteration - Value Function")
        viz.save_figure(fig_v, "vi_value_function.png", args.save_dir) 
        
        fig_p, _ = viz.plot_policy(policy, "Value Iteration - Policy")
        viz.save_figure(fig_p, "vi_policy.png", args.save_dir)
        
        fig_traj, _ = viz.plot_trajectory(trajectory, "Value Iteration - Trajectory")
        viz.save_figure(fig_traj, "vi_trajectory.png", args.save_dir)
        
        plt.show()
    
    env.close()


def run_policy_iteration(args, model):
    """Run Policy Iteration"""
    print("\n" + "=" * 60)
    print("Policy Iteration")
    print("=" * 60)
    
    env = GridWorldEnv(model=model)
    
    print("\nTraining Policy Iteration...")
    pi = PolicyIteration(model, gamma=args.gamma)
    V, policy = pi.train(max_iterations=100)
    
    agent = Agent(num_actions=model.num_actions, policy="learned", algorithm=pi)
    trajectory, reward, steps = run_episode(env, agent, max_steps=args.max_steps,
                                           verbose=args.verbose)
    
    print("\nPolicy Iteration Results:")
    print(f"Steps: {steps}, Total Reward: {reward:.1f}")
    
    if args.animate:
        run_animated_episode(args, model, pi, "Policy Iteration")

    if not args.no_viz:
        viz = GridWorldVisualizer(rows=args.rows, cols=args.cols,
                                 goal_states=model.goal_states,
                                 obstacles=model.obstacles,
                                 gamma=args.gamma)
                                 
        fig_v, _ = viz.plot_value_function(V, "Policy Iteration - Value Function")
        viz.save_figure(fig_v, "pi_value_function.png", args.save_dir) 
        
        fig_p, _ = viz.plot_policy(policy, "Policy Iteration - Policy")
        viz.save_figure(fig_p, "pi_policy.png", args.save_dir)
        
        fig_traj, _ = viz.plot_trajectory(trajectory, "Policy Iteration - Trajectory")
        viz.save_figure(fig_traj, "pi_trajectory.png", args.save_dir)
        
        plt.show()

    env.close()


def run_monte_carlo(args, model):
    """Run Monte Carlo"""
    print("\n" + "=" * 60)
    print("Monte Carlo Control")
    print("=" * 60)
    
    env = GridWorldEnv(model=model)
    
    print("\nTraining Monte Carlo...")
    mc = MonteCarlo(model, gamma=args.gamma, epsilon=args.epsilon)
    Q = mc.train(env, num_episodes=args.episodes, max_steps=args.max_steps)
    
    agent = Agent(num_actions=model.num_actions, policy="learned", algorithm=mc)
    trajectory, reward, steps = run_episode(env, agent, max_steps=args.max_steps,
                                           verbose=args.verbose)
    
    print("\nMonte Carlo Results:")
    print(f"Steps: {steps}, Total Reward: {reward:.1f}")
    
    if not args.no_viz:
        viz = GridWorldVisualizer(rows=args.rows, cols=args.cols,
                                 goal_states=model.goal_states,
                                 obstacles=model.obstacles)
        viz.plot_q_values(mc.Q, model, "Monte Carlo - Q-Values")
        viz.plot_trajectory(trajectory, "Monte Carlo - Trajectory")
        plt.show()
    
    env.close()


def run_q_learning(args, model):
    """Run Q-Learning"""
    print("\n" + "=" * 60)
    print("Q-Learning")
    print("=" * 60)
    
    env = GridWorldEnv(model=model)
    
    print("\nTraining Q-Learning...")
    ql = QLearning(model, alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon)
    Q, episode_rewards = ql.train(env, num_episodes=args.episodes, 
                                 max_steps=args.max_steps)
    
    agent = Agent(num_actions=model.num_actions, policy="learned", algorithm=ql)
    trajectory, reward, steps = run_episode(env, agent, max_steps=args.max_steps,
                                           verbose=args.verbose)
    
    print("\nQ-Learning Results:")
    print(f"Steps: {steps}, Total Reward: {reward:.1f}")
    
    if args.animate:
        run_animated_episode(args, model, ql, "Q-Learning") 
        
    if not args.no_viz:
        viz = GridWorldVisualizer(rows=args.rows, cols=args.cols,
                                 goal_states=model.goal_states,
                                 obstacles=model.obstacles,
                                 alpha=args.alpha,
                                 gamma=args.gamma,
                                 epsilon=args.epsilon)
        
        fig_q = viz.plot_q_values(ql.Q, model, "Q-Learning - Q-Values")
        viz.save_figure(fig_q, "ql_q_values.png", args.save_dir)
        
        fig_traj, _ = viz.plot_trajectory(trajectory, "Q-Learning - Trajectory")
        viz.save_figure(fig_traj, "ql_trajectory.png", args.save_dir)
        
        fig_lc = viz.plot_learning_curve(episode_rewards, "Q-Learning - Learning Curve")
        viz.save_figure(fig_lc, "ql_learning_curve.png", args.save_dir)
        
        plt.show()
    
    env.close()


def run_sarsa_lambda(args, model):
    """Run SARSA(lambda)"""
    print("\n" + "=" * 60)
    print(f"SARSA(lambda) (λ={args.lambda_val})")
    print("=" * 60)
    
    env = GridWorldEnv(model=model)
    
    print("\nTraining SARSA(λ)...")
    sl = SarsaLambda(model, alpha=args.alpha, gamma=args.gamma, 
                     epsilon=args.epsilon, lambda_val=args.lambda_val)
    Q, episode_rewards = sl.train(env, num_episodes=args.episodes, 
                                   max_steps=args.max_steps)
    
    agent = Agent(num_actions=model.num_actions, policy="learned", algorithm=sl)
    trajectory, reward, steps = run_episode(env, agent, max_steps=args.max_steps,
                                               verbose=args.verbose)
    
    print("\nSARSA(λ) Results:")
    print(f"Steps: {steps}, Total Reward: {reward:.1f}")
    
    if args.animate:
        run_animated_episode(args, model, sl, "SARSA(lambda)") 
        
    if not args.no_viz:
        viz = GridWorldVisualizer(rows=args.rows, cols=args.cols,
                                 goal_states=model.goal_states,
                                 obstacles=model.obstacles,
                                 alpha=args.alpha, 
                                 gamma=args.gamma, 
                                 epsilon=args.epsilon) 
        
        fig_q = viz.plot_q_values(sl.Q, model, f"SARSA(λ) - Q-Values (λ={args.lambda_val})")
        viz.save_figure(fig_q, "sl_q_values.png", args.save_dir) 
        
        fig_traj, _ = viz.plot_trajectory(trajectory, "SARSA(λ) - Trajectory")
        viz.save_figure(fig_traj, "sl_trajectory.png", args.save_dir) 
        
        fig_lc = viz.plot_learning_curve(episode_rewards, "SARSA(λ) - Learning Curve")
        viz.save_figure(fig_lc, "sl_learning_curve.png", args.save_dir) 
        
        plt.show()
    
    env.close()


def run_grid_size_experiment(args):
    """Run experiment comparing performance across different grid sizes"""
    print("\n" + "=" * 60)
    print("Grid Size Scaling Experiment")
    print("=" * 60)
    
    results = {
        'grid_sizes': [],
        'steps_data': [],
        'return_data': [],
        'training_times': []
    }
    
    for grid_size_str in args.grid_sizes:
        rows, cols = map(int, grid_size_str.split('x'))
        print(f"\n### Testing Grid Size: {rows}x{cols} ###")
        
        # Create model for this grid size
        model = GridWorldModel(
            rows=rows, cols=cols,
            start_states=[(0, 0)],
            goal_states=[(rows-1, cols-1)],
            obstacles=None,
            goal_reward=args.goal_reward,
            step_penalty=args.step_penalty,
            obstacle_penalty=args.obstacle_penalty
        )
        
        env = GridWorldEnv(model=model)
        
        # Train Q-Learning
        ql = QLearning(model, alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon)
        import time
        start_time = time.time()
        Q, episode_rewards = ql.train(env, num_episodes=args.episodes, 
                                     max_steps=args.max_steps)
        training_time = time.time() - start_time
        
        # Collect statistics
        results['grid_sizes'].append(grid_size_str)
        results['steps_data'].append(ql.episode_steps if hasattr(ql, 'episode_steps') else [])
        results['return_data'].append(episode_rewards)
        results['training_times'].append(training_time)
        
        print(f"Training time: {training_time:.2f}s")
        print(f"Final avg reward: {np.mean(episode_rewards[-50:]):.2f}")
        
        env.close()
    
    # Visualize results
    if not args.no_viz:
        viz = GridWorldVisualizer(rows=5, cols=5)
        
        # Steps vs grid size
        if all(results['steps_data']):
            fig_steps = viz.plot_steps_vs_grid_size(results['grid_sizes'], 
                                                    results['steps_data'],
                                                    "Steps per Episode vs Grid Size")
            viz.save_figure(fig_steps, "grid_size_steps.png", args.save_dir)
        
        # Returns vs grid size
        fig_returns = viz.plot_return_vs_grid_size(results['grid_sizes'],
                                                   results['return_data'],
                                                   "Average Return vs Grid Size")
        viz.save_figure(fig_returns, "grid_size_returns.png", args.save_dir)
        
        plt.show()
    
    # Save results
    if args.save_logs:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(args.save_dir, f"grid_experiment_{timestamp}.json")
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {
            'grid_sizes': results['grid_sizes'],
            'steps_data': [[int(s) for s in steps] for steps in results['steps_data']],
            'return_data': [[float(r) for r in returns] for returns in results['return_data']],
            'training_times': results['training_times']
        }
        
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\nExperiment results saved to: {results_path}")


def compare_all(args, model):
    """Compare all algorithms"""
    print("\n" + "=" * 60)
    print("Comparing All Algorithms")
    print("=" * 60)
    
    env = GridWorldEnv(model=model)
    results = {}
    
    # Value Iteration
    print("\n### Training Value Iteration ###")
    vi = ValueIteration(model, gamma=args.gamma)
    vi.train(max_iterations=1000)
    agent_vi = Agent(num_actions=model.num_actions, policy="learned", algorithm=vi)
    traj_vi, reward_vi, steps_vi = run_episode(env, agent_vi, max_steps=args.max_steps, 
                                               verbose=False)
    results['Value Iteration'] = {'trajectory': traj_vi, 'reward': reward_vi, 
                                  'steps': steps_vi}
    print(f"VI: {steps_vi} steps, reward: {reward_vi:.1f}")
    
    # Policy Iteration
    print("\n### Training Policy Iteration ###")
    pi = PolicyIteration(model, gamma=args.gamma)
    pi.train(max_iterations=100)
    agent_pi = Agent(num_actions=model.num_actions, policy="learned", algorithm=pi)
    traj_pi, reward_pi, steps_pi = run_episode(env, agent_pi, max_steps=args.max_steps,
                                               verbose=False)
    results['Policy Iteration'] = {'trajectory': traj_pi, 'reward': reward_pi, 
                                   'steps': steps_pi}
    print(f"PI: {steps_pi} steps, reward: {reward_pi:.1f}")
    
    # Monte Carlo
    print("\n### Training Monte Carlo ###")
    mc = MonteCarlo(model, gamma=args.gamma, epsilon=args.epsilon)
    mc.train(env, num_episodes=args.episodes, max_steps=args.max_steps)
    agent_mc = Agent(num_actions=model.num_actions, policy="learned", algorithm=mc)
    traj_mc, reward_mc, steps_mc = run_episode(env, agent_mc, max_steps=args.max_steps,
                                               verbose=False)
    results['Monte Carlo'] = {'trajectory': traj_mc, 'reward': reward_mc, 
                             'steps': steps_mc}
    print(f"MC: {steps_mc} steps, reward: {reward_mc:.1f}")
    
    # Q-Learning
    print("\n### Training Q-Learning ###")
    ql = QLearning(model, alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon)
    ql.train(env, num_episodes=args.episodes, max_steps=args.max_steps)
    agent_ql = Agent(num_actions=model.num_actions, policy="learned", algorithm=ql)
    traj_ql, reward_ql, steps_ql = run_episode(env, agent_ql, max_steps=args.max_steps,
                                               verbose=False)
    results['Q-Learning'] = {'trajectory': traj_ql, 'reward': reward_ql, 
                            'steps': steps_ql}
    print(f"QL: {steps_ql} steps, reward: {reward_ql:.1f}")
    
    # DQL
    print("\n### Training Deep Q-Learning ###")
    dqn = DQN(model=model, state_dim=2, hidden_dims=args.dqn_hidden,
              alpha=args.alpha, gamma=args.gamma,
              epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=args.epsilon_decay,
              buffer_size=args.dqn_buffer_size, batch_size=args.dqn_batch_size,
              target_update_freq=args.dqn_target_update)
    dqn.train(env, num_episodes=args.episodes, max_steps=args.max_steps, verbose_freq=100)
    agent_dqn = Agent(num_actions=model.num_actions, policy="learned", algorithm=dqn)
    traj_dqn, reward_dqn, steps_dqn = run_episode(env, agent_dqn, max_steps=args.max_steps,
                                                  verbose=False)
    results['Deep Q-Learning'] = {'trajectory': traj_dqn, 'reward': reward_dqn,
                                 'steps': steps_dqn}
    print(f"DQL: {steps_dqn} steps, reward: {reward_dqn:.1f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, data in results.items():
        print(f"{name:20s}: {data['steps']:2d} steps, reward: {data['reward']:6.1f}")
    
    if not args.no_viz:
        viz = GridWorldVisualizer(rows=args.rows, cols=args.cols,
                                 goal_states=model.goal_states,
                                 obstacles=model.obstacles)
        fig, _ = viz.plot_comparison(results, "Algorithm Comparison")
        viz.save_figure(fig, "algorithm_comparison.png", args.save_dir)
        plt.show()
    
    env.close()


def load_and_visualize_logs(args):
    """Load training logs and generate visualizations"""
    print(f"\nLoading training logs from: {args.load_logs}")
    
    viz = GridWorldVisualizer(rows=args.rows, cols=args.cols)
    figures = viz.load_and_visualize_log(args.load_logs, save_dir=args.save_dir)
    
    if not args.no_viz:
        plt.show()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    
    print("=" * 60)
    print("GridWorld: Reinforcement Learning Framework")
    print("=" * 60)
    
    # Load and visualize logs if specified
    if args.load_logs:
        load_and_visualize_logs(args)
        return
    
    # Run grid size experiment if specified
    if args.grid_size_experiment:
        run_grid_size_experiment(args)
        return
    
    # Create model
    model = create_model(args)
    print(f"\n{model}")
    
    # Run selected algorithm
    algorithms = {
        'random': run_random,
        'vi': run_value_iteration,
        'pi': run_policy_iteration,
        'mc': run_monte_carlo,
        'ql': run_q_learning,
        'sl': run_sarsa_lambda,
        'dql': run_dql,
        'compare': compare_all
    }
    
    algorithms[args.algorithm](args, model)
    
    print("\n" + "=" * 60)
    print("Execution Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()