from model import GridWorldModel
from agent import Agent
from environment import GridWorldEnv
from visualizer import GridWorldVisualizer, visualize_episode
import matplotlib.pyplot as plt
import numpy as np

def run_episode(env, agent, max_steps=50, verbose=True):
    state, info = env.reset(seed=42)
    agent.reset()
    
    if verbose:
        print("Starting episode...")
        env.render()
    
    for step in range(max_steps):
        action = agent.select_action(state)
        
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
                print(f"\n Goal reached in {agent.steps} steps!")
                print(f"Total reward: {agent.total_reward:.1f}")
            break
    
    return env.get_trajectory(), agent.total_reward, agent.steps


def main():
    # Create environment
    print("=" * 50)
    print("GridWorld: Agent-Model-Environment Simulation")
    print("=" * 50)
    
    env = GridWorldEnv(rows=5, cols=5, render_mode="human")
    
    # Test with random agent
    print("\n### Random Agent ###")
    agent_random = Agent(num_actions=env.model.num_actions, policy="random")
    trajectory_random, reward, steps = run_episode(env, agent_random, max_steps=50)
    
    env.close()
    
    # Create visualizer
    viz = GridWorldVisualizer(rows=5, cols=5)
    
    # Visualize random agent trajectory
    print(f"\nRandom agent trajectory length: {len(trajectory_random)} steps")
    fig1, ax1 = viz.plot_trajectory(trajectory_random, "Random Agent Trajectory")
    
    plt.show()
    
    print("" + "=" * 50)


if __name__ == "__main__":
    main()