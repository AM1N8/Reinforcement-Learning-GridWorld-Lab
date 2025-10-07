from environment import GridWorldEnv

# 1. Test the default mode (Non-dynamic)
# Features should be fixed (default start=(0,0), goal=(4,4) for a 5x5)
env_static = GridWorldEnv(rows=5, cols=5, render_mode="human", dynamic_mode=False)
print("--- Static Mode Reset 1 ---")
env_static.reset(seed=1)
print(f"Goal: {env_static.model.goal_states}, Obstacles: {len(env_static.model.obstacles)}")

print("\n--- Static Mode Reset 2 ---")
env_static.reset(seed=2)
# Goal and obstacles should be the same as Reset 1
print(f"Goal: {env_static.model.goal_states}, Obstacles: {len(env_static.model.obstacles)}") 

# 2. Test the dynamic mode
# Goal and obstacles should change on every reset
env_dynamic = GridWorldEnv(rows=8, cols=8, render_mode="human", 
                           dynamic_mode=True, num_dynamic_obstacles=8)

print("\n" + "="*20)
print("--- Dynamic Mode Reset 1 ---")
env_dynamic.reset(seed=10)
print(f"Goal: {env_dynamic.model.goal_states}, Obstacles: {len(env_dynamic.model.obstacles)}")
# The rendered output should show a new 'G' and new 'X's

print("\n--- Dynamic Mode Reset 2 ---")
env_dynamic.reset(seed=20)
# Goal and obstacles should be different from Reset 1
print(f"Goal: {env_dynamic.model.goal_states}, Obstacles: {len(env_dynamic.model.obstacles)}")
# The rendered output should show yet another configuration