# grid_world.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to University of California, Riverside and the authors.
# 
# Authors: Pei Xu (peixu@stanford.edu) and Ioannis Karamouzas (ioannis@cs.ucr.edu)
"""
The package `matplotlib` is needed for the program to run.

The Grid World environment has discrete state and action spaces
and allows for both model-based and model-free access.

It has the following properties:
    env.observation_space.n     # the number of states
    env.action_space.n          # the number of actions
    env.trans_model             # the transition/dynamics model

In value_iteration and policy_iteration, you can access the transition model 
at a given state s and action by calling
    t = env.trans_model[s][a]
where s is an integer in the range [0, env.observation_space.n),
      a is an integer in the range [0, env.action_space.n), and
      t is a list of four-element tuples in the form of
        (p, s_, r, terminal)
where s_ is a new state reachable from the state s by taking the action a,
      p is the probability to reach s_ from s by a, i.e. p(s_|s, a),
      r is the reward of reaching s_ from s by a, and
      terminal is a boolean flag to indicate if s_ is a terminal state.

In q_learning, once a terminal state is reached, 
the environment should be (re)initialized by
    s = env.reset()
where s is the initial state.
An experience (sample) can be collected from s by taking an action a as follows:
    s_, r, terminal, info = env.step(a)
where s_ is the resulted state by taking the action a,
      r is the reward achieved by taking the action a,
      terminal is a boolean flag to indicate if s_ is a terminal state, and
      info is just used to keep compatible with openAI gym library.


A Logger instance is provided for each function, through which you can
visualize the process of the algorithm.
You can visualize the value, v, and policy, pi, for the i-th iteration by
    logger.log(i, v, pi)
You can also only update the visualization of the v values by
    logger.log(i, v)
"""


# use random library if needed
import random


def value_iteration(env, gamma, max_iterations, logger):
    """
    Implement value iteration to return a deterministic policy for all states.
    See lines 20-30 for details.  

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the reward discount factor
    max_iterations: integer
        the maximum number of value iterations that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
        Hint: The value iteration may converge before reaching max_iterations.  
        In this case, you want to exit the algorithm earlier. A way to check 
        if value iteration has already converged is to check whether 
        the infinity norm between the values before and after an iteration is small enough. 
        In the gridworld environments, 1e-4 (theta parameter in the pseudocode) is an acceptable tolerance.
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    TRANSITION_MODEL = env.trans_model

    v = [0] * NUM_STATES
    pi = [0] * NUM_STATES
    # Visualize the value and policy 
    logger.log(0, v, pi)
    # At each iteration, you may need to keep track of pi to perform logging
   
### Please finish the code below ##############################################
###############################################################################
    theta = 1e-4
    
    for k in range(max_iterations + 1):
        v_new = v.copy()
        delta = 0
        
        for s in range(NUM_STATES):
            actionValues = []
            for a in range(NUM_ACTIONS):
                total = 0
                for (p, s_, r, terminal) in TRANSITION_MODEL[s][a]:
                    total += p * (r + gamma * v[s_])
                actionValues.append(total) 


            # choose the max out of all the bellman sum inside the action_value array
            maxValue = max(actionValues)
            v_new[s] = maxValue

            delta = max(delta, abs(maxValue - v[s]))

        v = v_new

        # update policy
        for s in range(NUM_STATES):
            optimalAction = 0
            optimalValue = float('-inf')
            for a in range(NUM_ACTIONS):
                total = 0
                for (p, s_, r, terminal) in TRANSITION_MODEL[s][a]:
                    total += p * (r + gamma * v[s_])
                if total > optimalValue:
                    optimalValue = total
                    optimalAction = a
            pi[s] = optimalAction

        logger.log(k + 1, v, pi)
        # bellman update/back-up
        if delta < theta: 
            break
###############################################################################
    return pi


def policy_iteration(env, gamma, max_iterations, logger):
    """
    Optional: Implement policy iteration to return a deterministic policy for all states.
    See lines 20-30 for details.  

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the reward discount factor
    max_iterations: integer
        the maximum number of policy iterations that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
        Hint 1: Policy iteration may converge before reaching max_iterations. 
        In this case, you should exit the algorithm. A simple way to check 
        if the algorithm has already converged is by simply checking whether
        the policy at each state hasn't changed from the previous iteration.
        Hint 2: The value iteration during policy evaluation usually converges 
        very fast and policy evaluation should end upon convergence. A way to check 
        if policy evaluation has converged is to check whether the infinity norm 
        norm between the values before and after an iteration is small enough. 
        In the gridworld environments, 1e-4 is an acceptable tolerance.
    logger: app.grid_world.App.Logger
        a logger instance to record and visualize the iteration process.
        During policy evaluation, the V-values will be updated without changing the current policy; 
        here you can update the visualization of values by simply calling logger.log(i, v).
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    TRANSITION_MODEL = env.trans_model
    
    v = [0.0] * NUM_STATES
    pi = [random.randint(0, NUM_ACTIONS-1)] * NUM_STATES
    # Visualize the initial value and policy
    logger.log(0, v, pi)

### Please finish the code below ##############################################
###############################################################################
    theta = 1e-4
    converge = True
    for k in range(max_iterations + 1):
        while converge:
            v_new = v.copy()
            delta = 0
            for s in range(NUM_STATES):
                total = 0
                for (p, s_, r, terminal) in TRANSITION_MODEL[s][pi[s]]: #transition for pi[s] action
                    total += p * (r + gamma * v[s_])
                v_new[s] = total 
                delta = max(delta, abs(v_new[s] - v[s]))
            v = v_new #do this when all states are processed
            if delta < theta: #check to see if it converged. if so then exit
                break
        # policy improvement
        policyStable = True
        for s in range(NUM_STATES):
            oldAction = pi[s]
            actionValues = []
            for a in range(NUM_ACTIONS):
                total = 0
                for (p, s_, r, terminal) in TRANSITION_MODEL[s][a]:
                    total += p * (r + gamma * v[s_])
                actionValues.append(total) #cummulating all of the actions into an array
            pi[s] = actionValues.index(max(actionValues)) #returns the action with the highest value(best one)      
            if oldAction != pi[s]:
                policyStable = False
        logger.log(k + 1, v, pi)
        if policyStable:
            break
###############################################################################
    return pi


def q_learning(env, gamma, max_iterations, logger):
    """
    Implement Q-learning to return a deterministic policy for all states.
    Please note that in RL you do not have access to the transition model 
    and the reward function, i.e. you cannot call env.trans_model. 
    Instead you need to learn policies by collecting samples using env.step
    See lines 32-42 for more details. 

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the discount factor
    max_iterations: integer
        the maximum number of iterations (either training episodes or total steps) that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process.
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n

    v = [0] * NUM_STATES
    pi = [0] * NUM_STATES
    # Visualize the initial value and policy
    logger.log(0, v, pi)

    #########################
    # Adjust superparameters as you see fit
    #
    # parameter for the epsilon-greedy method to trade off exploration and exploitation
    eps = 1
    # learning rate for updating q values based on sample estimates
    alpha = 0.1
    #########################

    

### Please finish the code below ##############################################
###############################################################################
    qTable = [[0] * NUM_ACTIONS for _ in range((NUM_STATES))]
    minEps = 0.1
    totalSteps = 0 #stop once total steps reaches the max iteration
    s = env.reset() #resetting s after each episode
    while totalSteps < max_iterations:
        while True: #in instruction max iteration represent the number of steps and not total number of episodes
            epsilonStrink = (eps - minEps) * (totalSteps / max_iterations) #represents how much epsilon should have shrunk so far
            epsilon = max(minEps, eps - epsilonStrink) #getting the max of linear decayed epsilon versus the minimum epsilon
            if random.random() < epsilon: #want to always explore when epsilon is high
                a = random.randint(0, NUM_ACTIONS-1) 
            else: #if eps is low choose best action
                a = qTable[s].index(max((qTable[s]))) #returns the index of the action with highest q-value based on the state
            s_, r, terminal, info = env.step(a)
            if terminal == True:
                target = r
            else:
                target = r + gamma * max(qTable[s_]) #gets max q-value for all possible actions in the next state s_
            qTable[s][a] = (1 - alpha) * qTable[s][a] + alpha * target
            s = s_
            totalSteps += 1
            v = [max(qTable[s]) for s in range(NUM_STATES)] #look through all states and return max Q-value among all the actions
            pi = [qTable[s].index(max((qTable[s]))) for s in range(NUM_STATES)] #look through all states and return index/action with highest Q-value
            logger.log(totalSteps, v, pi) #records iteration process
            if terminal or totalSteps >= max_iterations:
                break   # Just break. Reset comes outside inner while loop.
        s = env.reset()
###############################################################################
    return pi



if __name__ == "__main__":
    from app.grid_world import App
    import tkinter as tk

    algs = {
        "Value Iteration": value_iteration,
        "Policy Iteration": policy_iteration,
        "Q-Learning": q_learning
   }
    worlds = {
        # o for obstacle
        # s for start cell
        "world1": App.DEFAULT_WORLD,
        "world2": lambda : [
            ["_", "_", "_", "_", "_"],
            ["s", "_", "_", "_", 1],
            [-100, -100, -100, -100, -100],
        ],
        "world3": lambda : [
            ["_", "_", "_", "_", "_"],
            ["_", "o", "_", "_", "_"],
            ["_", "o",   1, "_",  10],
            ["s", "_", "_", "_", "_"],
            [-10, -10, -10, -10, -10]
        ]
    }

    root = tk.Tk()
    App(algs, worlds, root)
    tk.mainloop()