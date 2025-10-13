import mdp, util
from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        A ValueIterationAgent takes an MDP on initialization and runs value iteration
        for a given number of iterations using the supplied discount factor.
    """
    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount=0.9, iterations=100):
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        for i in range(self.iterations):
            newValues = util.Counter()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    newValues[state] = 0
                    continue
                actions = self.mdp.getPossibleActions(state)
                if not actions:
                    newValues[state] = 0
                    continue
                q_values = [self.computeQValueFromValues(state, a) for a in actions]
                newValues[state] = max(q_values)
            self.values = newValues

    def getValue(self, state):
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        q_value = 0
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, nextState)
            q_value += prob * (reward + self.discount * self.values[nextState])
        return q_value

    def computeActionFromValues(self, state):
        if self.mdp.isTerminal(state):
            return None
        actions = self.mdp.getPossibleActions(state)
        if not actions:
            return None
        best_action = None
        best_value = float('-inf')
        for action in actions:
            q_value = self.computeQValueFromValues(state, action)
            if q_value > best_value:
                best_value = q_value
                best_action = action
        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
