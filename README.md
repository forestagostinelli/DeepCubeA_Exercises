# DeepCubeA Exercises
These are exercises to understand the 
[DeepCubeA](https://cse.sc.edu/~foresta/assets/files/SolvingTheRubiksCubeWithDeepReinforcementLearningAndSearch_Final.pdf) 
algorithm.

These exercises are for anyone who is getting started with deep reinforcement learning and search.
The goal of these exercises is to implement a method that learns to solve the 8-puzzle.
The solutions to these exercises can be run on a standard laptop CPU in less than 10 minutes.
Sample outputs of solutions to each exercise are also provided in `sample_outputs/`.

This currently contains one exercise. More to come.

For any issues, please contact Forest Agostinelli (foresta@cse.sc.edu)

# Setup
These exercises require Python3, PyTorch, and numpy.

# Exercise 1: Supervised Learning
We would like to build a DNN that learns to approximate the cost-to-go from any state of the 8-puzzle to the 
goal state (the configuration of the solution). This also corresponds to the minimum number of moves required to 
solve the state.

For this exercise, there is an oracle that can tell us the cost-to-go for any state. All we have to do is design a DNN
architecture that can map any 8-puzzle state to its estimated cost-to-go.

To complete this exercise, you will have to implement:
- `get_nnet_model` in `to_implement/functions.py`
    - This method returns the pytorch model (torch.nn.Module) that maps any 8-puzzle state to its cost-to-go.
    - The dimensionality of the input will be (B x 81), where B is the batch size. This is because the 8-puzzle has 9 tiles, including the blank tile. 
    The representation given to the neural network is a one-hot representation for each tile. The dimensionality of the output will be (B x 1).
- `train_nnet` in `to_implement/functions.py`
    - This method trains the pytorch model
    
# Exercise 2: Approximate Value Iteration
The assumption of having an oracle is too strong for most real-world applications.
Therefore, we turn to value iteration which allows us to approximate the cost-to-go from scratch.
In this setting, we start with a naive cost-to-go (a randomly initialized neural network) and use the Bellman equation to obtain an updated estimate of the cost-to-go.
We then train the neural network to match this updated estimate. We iterate this process, bootstrapping from previous estimates to obtain updated estimates.

Traditional value iteration stores all possible states and their corresponding cost-to-go in a table and updates them using the Bellman equation.
However, we cannot store all possible states in a table for environments with large state spaces.
Therefore, we turn to approximate value iteration and use a neural network to approximate the cost-to-go.

Value iteration is outlined in chapter 4 of Sutton and Barto's (S&B) [reinforcement learning book](http://www.incompleteideas.net/book/RLbook2020.pdf).
One key difference is that, in S&B, they are maximizing value, while in this context, we are minimizing cost.
See the DeepCubeA paper for this version of value iteration.

For speed considerations, it is important to batch data sent to the neural network instead of doing it in a loop.
The functions `misc_utils.flatten` and `misc_utils.unflatten` may help you with this when implementing value iteration.

Use `env.state_to_nnet_input(states)` to convert a list of states to their numpy representation.
Use `env.expand(states)` to get a list of next states for each state as well as the transition costs