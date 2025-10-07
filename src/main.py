from model import GridWorldModel
from agent import Agent
from environment import GridWorldEnv
from visualizer import GridWorldVisualizer, visualize_episode
from algorithms import ValueIteration, PolicyIteration, MonteCarlo, QLearning
import matplotlib.pyplot as plt
import numpy as np

seed = 123
np.random.seed(seed)

def run_episode(env, agent, max_steps=50, verbose=True):
    """Run a single episode with the given agent"""
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


def test_random_agent():
    """Test basic random agent"""
    print("\n" + "=" * 60)
    print("Testing Random Agent")
    print("=" * 60)
    
    env = GridWorldEnv(rows=5, cols=5, render_mode="human")
    agent_random = Agent(num_actions=env.model.num_actions, policy="random")
    trajectory_random, reward, steps = run_episode(env, agent_random, max_steps=50)
    env.close()
    
    # Visualize
    viz = GridWorldVisualizer(rows=5, cols=5)
    fig, ax = viz.plot_trajectory(trajectory_random, "Random Agent Trajectory")
    plt.show()
    
    return trajectory_random


def test_value_iteration():
    """Test Value Iteration algorithm"""
    print("\n" + "=" * 60)
    print("Value Iteration")
    print("=" * 60)
    
    env = GridWorldEnv(rows=4, cols=4)
    
    # Train Value Iteration
    print("\nTraining Value Iteration...")
    vi = ValueIteration(env.model, gamma=0.99, theta=1e-6)
    V, policy = vi.train(max_iterations=1000)
    
    # Test the learned policy
    print("\nTesting learned policy...")
    agent_vi = Agent(num_actions=env.model.num_actions, policy="learned", algorithm=vi)
    trajectory, reward, steps = run_episode(env, agent_vi, max_steps=50, verbose=True)
    
    print(f"\nValue Iteration Results:")
    print(f"Steps to goal: {steps}")
    print(f"Total reward: {reward:.1f}")
    
    # Visualize
    viz = GridWorldVisualizer(rows=4, cols=4)
    
    fig1, ax1 = viz.plot_value_function(V, "Value Iteration - Value Function")
    fig2, ax2 = viz.plot_policy(policy, "Value Iteration - Learned Policy")
    fig3, ax3 = viz.plot_trajectory(trajectory, "Value Iteration - Trajectory")
    
    plt.show()
    env.close()
    
    return vi, trajectory


def test_policy_iteration():
    """Test Policy Iteration algorithm"""
    print("\n" + "=" * 60)
    print("Policy Iteration")
    print("=" * 60)
    
    env = GridWorldEnv(rows=5, cols=5)
    
    # Train Policy Iteration
    print("\nTraining Policy Iteration...")
    pi = PolicyIteration(env.model, gamma=0.99, theta=1e-6)
    V, policy = pi.train(max_iterations=100)
    
    # Test the learned policy
    print("\nTesting learned policy...")
    agent_pi = Agent(num_actions=env.model.num_actions, policy="learned", algorithm=pi)
    trajectory, reward, steps = run_episode(env, agent_pi, max_steps=50, verbose=True)
    
    print(f"\nPolicy Iteration Results:")
    print(f"Steps to goal: {steps}")
    print(f"Total reward: {reward:.1f}")
    
    # Visualize
    viz = GridWorldVisualizer(rows=5, cols=5)
    
    fig1, ax1 = viz.plot_value_function(V, "Policy Iteration - Value Function")
    fig2, ax2 = viz.plot_policy(policy, "Policy Iteration - Learned Policy")
    fig3, ax3 = viz.plot_trajectory(trajectory, "Policy Iteration - Trajectory")
    
    plt.show()
    env.close()
    
    return pi, trajectory


def test_monte_carlo():
    """Test Monte Carlo algorithm"""
    print("\n" + "=" * 60)
    print("Monte Carlo Control")
    print("=" * 60)
    
    env = GridWorldEnv(rows=5, cols=5)
    
    # Train Monte Carlo
    print("\nTraining Monte Carlo...")
    mc = MonteCarlo(env.model, gamma=0.99, epsilon=0.1)
    Q = mc.train(env, num_episodes=1000, max_steps=100)
    
    # Test the learned policy
    print("\nTesting learned policy...")
    agent_mc = Agent(num_actions=env.model.num_actions, policy="learned", algorithm=mc)
    trajectory, reward, steps = run_episode(env, agent_mc, max_steps=50, verbose=True)
    
    print(f"\nMonte Carlo Results:")
    print(f"Steps to goal: {steps}")
    print(f"Total reward: {reward:.1f}")
    
    # Visualize
    viz = GridWorldVisualizer(rows=5, cols=5)
    
    fig1, ax1 = viz.plot_q_values(mc.Q, env.model, "Monte Carlo - Q-Values")
    fig2, ax2 = viz.plot_trajectory(trajectory, "Monte Carlo - Trajectory")
    
    plt.show()
    env.close()
    
    return mc, trajectory


def test_q_learning():
    """Test Q-Learning algorithm"""
    print("\n" + "=" * 60)
    print("Q-Learning")
    print("=" * 60)
    
    env = GridWorldEnv(rows=4, cols=4)
    
    # Train Q-Learning
    print("\nTraining Q-Learning...")
    ql = QLearning(env.model, alpha=0.1, gamma=0.99, epsilon=0.05)
    Q, episode_rewards = ql.train(env, num_episodes=500, max_steps=100)
    
    # Test the learned policy
    print("\nTesting learned policy...")
    agent_ql = Agent(num_actions=env.model.num_actions, policy="learned", algorithm=ql)
    trajectory, reward, steps = run_episode(env, agent_ql, max_steps=50, verbose=True)
    
    print(f"\nQ-Learning Results:")
    print(f"Steps to goal: {steps}")
    print(f"Total reward: {reward:.1f}")
    
    # Visualize
    viz = GridWorldVisualizer(rows=4, cols=4)
    
    fig1, ax1 = viz.plot_q_values(ql.Q, env.model, "Q-Learning - Q-Values")
    fig2, ax2 = viz.plot_trajectory(trajectory, "Q-Learning - Trajectory")
    fig3, ax3 = viz.plot_learning_curve(episode_rewards, "Q-Learning - Learning Curve")
    
    plt.show()
    env.close()
    
    return ql, trajectory


def compare_all_algorithms():
    """Compare all RL algorithms side by side"""
    print("\n" + "=" * 60)
    print("Comparing All Algorithms")
    print("=" * 60)
    
    env = GridWorldEnv(rows=5, cols=5)
    results = {}
    
    # Value Iteration
    print("\n### Training Value Iteration ###")
    vi = ValueIteration(env.model, gamma=0.99)
    vi.train(max_iterations=1000)
    agent_vi = Agent(num_actions=env.model.num_actions, policy="learned", algorithm=vi)
    traj_vi, reward_vi, steps_vi = run_episode(env, agent_vi, max_steps=50, verbose=False)
    results['Value Iteration'] = {'trajectory': traj_vi, 'reward': reward_vi, 'steps': steps_vi}
    print(f"VI: {steps_vi} steps, reward: {reward_vi:.1f}")
    
    # Policy Iteration
    print("\n### Training Policy Iteration ###")
    pi = PolicyIteration(env.model, gamma=0.99)
    pi.train(max_iterations=100)
    agent_pi = Agent(num_actions=env.model.num_actions, policy="learned", algorithm=pi)
    traj_pi, reward_pi, steps_pi = run_episode(env, agent_pi, max_steps=50, verbose=False)
    results['Policy Iteration'] = {'trajectory': traj_pi, 'reward': reward_pi, 'steps': steps_pi}
    print(f"PI: {steps_pi} steps, reward: {reward_pi:.1f}")
    
    # Monte Carlo
    print("\n### Training Monte Carlo ###")
    mc = MonteCarlo(env.model, gamma=0.99, epsilon=0.1)
    mc.train(env, num_episodes=500, max_steps=100)
    agent_mc = Agent(num_actions=env.model.num_actions, policy="learned", algorithm=mc)
    traj_mc, reward_mc, steps_mc = run_episode(env, agent_mc, max_steps=50, verbose=False)
    results['Monte Carlo'] = {'trajectory': traj_mc, 'reward': reward_mc, 'steps': steps_mc}
    print(f"MC: {steps_mc} steps, reward: {reward_mc:.1f}")
    
    # Q-Learning
    print("\n### Training Q-Learning ###")
    ql = QLearning(env.model, alpha=0.1, gamma=0.99, epsilon=0.1)
    ql.train(env, num_episodes=500, max_steps=100)
    agent_ql = Agent(num_actions=env.model.num_actions, policy="learned", algorithm=ql)
    traj_ql, reward_ql, steps_ql = run_episode(env, agent_ql, max_steps=50, verbose=False)
    results['Q-Learning'] = {'trajectory': traj_ql, 'reward': reward_ql, 'steps': steps_ql}
    print(f"QL: {steps_ql} steps, reward: {reward_ql:.1f}")
    
    # Visualize comparison
    viz = GridWorldVisualizer(rows=4, cols=4)
    fig, axes = viz.plot_comparison(results, "Algorithm Comparison")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, data in results.items():
        print(f"{name:20s}: {data['steps']:2d} steps, reward: {data['reward']:6.1f}")
    
    plt.show()
    env.close()
    
    return results


def main():
    print("=" * 60)
    print("GridWorld: Reinforcement Learning Algorithms")
    print("=" * 60)
    
    # Uncomment the test you want to run:
    
    # Test individual algorithms:
    # test_random_agent()
    # test_value_iteration()
    # test_policy_iteration()
    # test_monte_carlo()
    test_q_learning()
    
    # Compare all algorithms:
    # compare_all_algorithms()


if __name__ == "__main__":
    main()