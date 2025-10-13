# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
from backend import ReplayMemory

import backend
import gridworld


import random,util,math
import numpy as np
import copy

class QLearningAgent(ReinforcementAgent):
    def __init__(self, **args):
        ReinforcementAgent.__init__(self, **args)
        self.qValues = util.Counter()  # Q-values stored as a dictionary

    def getQValue(self, state, action):
        return self.qValues[(state, action)]

    def computeValueFromQValues(self, state):
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return 0.0
        return max(self.getQValue(state, a) for a in legalActions)

    def computeActionFromQValues(self, state):
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None
        best_value = float('-inf')
        best_actions = []
        for a in legalActions:
            q = self.getQValue(state, a)
            if q > best_value:
                best_value = q
                best_actions = [a]
            elif q == best_value:
                best_actions.append(a)
        return random.choice(best_actions)

    def getAction(self, state):
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        else:
            return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward: float):
        sample = reward + self.discount * self.computeValueFromQValues(nextState)
        self.qValues[(state, action)] = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * sample



class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action

class ApproximateQAgent(PacmanQAgent):
    """
    ApproximateQLearningAgent
    
    You should only have to overwrite getQValue, update, __init__
    and possibly final (if you want to do something when training ends).
    All other QLearningAgent functions should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
        Should return Q(state,action) = w * featureVector
        where * is the dotProduct operator
        """
        features = self.featExtractor.getFeatures(state, action)
        q_value = 0.0
        for f, v in features.items():
            q_value += self.weights[f] * v
        return q_value

    def update(self, state, action, nextState, reward: float):
        """
        Should update your weights based on transition
        """
        features = self.featExtractor.getFeatures(state, action)
        difference = (reward + self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(state, action)
        
        # Update each weight
        for f, v in features.items():
            self.weights[f] += self.alpha * difference * v

    def final(self, state):
        """Called at the end of each game."""
        # Call the super-class final method
        PacmanQAgent.final(self, state)

        # Did we finish training?
        if self.episodesSoFar == self.numTraining:
            # You might want to print your weights here for debugging
            # print("Final weights:", self.weights)
            pass