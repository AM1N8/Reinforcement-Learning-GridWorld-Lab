# to run this project i am using uv instead of pip :

1. pip install uv 
2. uv sync 
3. uv run main.py

- main.py has functions to run ( VI PI MC and Qlearning ) and compare the algorithms 
- algorithms.py has all the classes for algorithms ( GPI MC QLearning)
- it would have been better to use arg parsers in main.py but i couldn't figure out how to do it 

# Qlearning results : 
```
Training Q-Learning...
Episode 100/500, Avg Reward: 2.19
Episode 200/500, Avg Reward: 4.67
Episode 300/500, Avg Reward: 4.87
Episode 400/500, Avg Reward: 4.71
Episode 500/500, Avg Reward: 4.71

Testing learned policy...
Starting episode...

Step 1: Action = RIGHT
Position: [0 1], Reward: -1.0, Total: -1.0

Step 2: Action = DOWN
Position: [1 1], Reward: -1.0, Total: -2.0

Step 3: Action = RIGHT
Position: [1 2], Reward: -1.0, Total: -3.0

Step 4: Action = DOWN
Position: [2 2], Reward: -1.0, Total: -4.0

Step 5: Action = DOWN
Position: [3 2], Reward: -1.0, Total: -5.0

Step 6: Action = RIGHT
Position: [3 3], Reward: 10.0, Total: 5.0

Goal reached in 6 steps!
Total reward: 5.0

Q-Learning Results:
Steps to goal: 6
Total reward: 5.0
```