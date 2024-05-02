# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        max_score = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == max_score]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        newFood_list = newFood.asList()

        dis_closest_food = min([util.manhattanDistance(newPos, foodPos) for foodPos in newFood_list], default=99999999)
        dis_closest_ghost = min([util.manhattanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates], default=99999999)

        "ghost is very close to pacman"
        ghost_score = -1000 if dis_closest_ghost <= 2 else 0
        
        "eat scaredghost"
        scared_ghost_score = 99999999 if dis_closest_ghost <= newScaredTimes[0] else 0

        "reciprocal of distance to food"
        food_score = 10/dis_closest_food if dis_closest_food != 0 else 0

        final_score = successorGameState.getScore() + food_score + ghost_score + scared_ghost_score
        return final_score


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # Minimax implementation with depth increasing for each agent's action, reflecting sequential turns in gameplay.
        def minimax(gameState, agentIndex, depth):
            if gameState.isWin() or gameState.isLose() or depth == self.depth * gameState.getNumAgents():
                return self.evaluationFunction(gameState)
            if agentIndex == 0:  # Pacman
                return maxValue(gameState, agentIndex, depth)
            else:  # Ghost
                return minValue(gameState, agentIndex, depth)

        def maxValue(gameState, agentIndex, depth):
            v = float("-inf")  # initialize
            for action in gameState.getLegalActions(agentIndex):  # for action in list
                successor = gameState.generateSuccessor(agentIndex, action)  # successor game state after an agent takes an action
                next_agentIndex = (agentIndex + 1) % gameState.getNumAgents()
                v = max(v, minimax(successor, next_agentIndex, depth + 1))  # recursively call minimax function
            return v

        def minValue(gameState, agentIndex, depth):
            v = float("inf")
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                next_agentIndex = (agentIndex + 1) % gameState.getNumAgents()
                v = min(v, minimax(successor, next_agentIndex, depth + 1))
            return v

        ideal_action = max(gameState.getLegalActions(0), key=lambda action: minimax(gameState.generateSuccessor(0, action), 1, 1))
        return ideal_action





class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        "*** YOUR CODE HERE ***"
        # this is similar to the minimax function above
        # Alpha-Beta Pruning with decreasing depth at each agent's turn, ending search when depth reaches zero.
        def alphabetapruning(gameState, agentIndex, depth, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            if agentIndex == 0:  # Pacman (Maximizer)
                return maxValue(gameState, agentIndex, depth, alpha, beta)
            else:  # Ghosts (Minimizer)
                return minValue(gameState, agentIndex, depth, alpha, beta)
        
        def maxValue(gameState, agentIndex, depth, alpha, beta):
            v = float("-inf")
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                v = max(v, alphabetapruning(successor, 1, depth, alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v
        
        
        def minValue(gameState, agentIndex, depth, alpha, beta):
            v = float("inf")
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                next_agentIndex = (agentIndex + 1) % gameState.getNumAgents()
                next_depth = depth - 1 if next_agentIndex == 0 else depth
                v = min(v, alphabetapruning(successor, next_agentIndex, next_depth, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v
        
        
        alpha = float("-inf")
        beta = float("inf")
        max_score = float("-inf")
        for action in gameState.getLegalActions(0):
            score = alphabetapruning(gameState.generateSuccessor(0, action), 1, self.depth, alpha, beta)
            if score > max_score:
                max_score, ideal_action = score, action
            alpha = max(alpha, score)
        return ideal_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # In a method similar to AlphaBeta pruning, the only modification required is the creation of the expValue function.
        def expectimax(gameState, agentIndex, depth):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            if agentIndex == 0:
                return maxValue(gameState, agentIndex, depth)
            else:
                return expValue(gameState, agentIndex, depth)

        def maxValue(gameState, agentIndex, depth):
            v = float("-inf")
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                v = max(v,expectimax(successor, 1, depth))
            return v

        def expValue(gameState, agentIndex, depth):
            v = 0.0
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                next_agentIndex = (agentIndex + 1) % gameState.getNumAgents()
                next_depth = depth - 1 if next_agentIndex == 0 else depth
                v += expectimax(successor, next_agentIndex, next_depth) / len(gameState.getLegalActions(agentIndex))
            return v
        
        max_score = float("-inf")
        for action in gameState.getLegalActions(0):
            score = expectimax(gameState.generateSuccessor(0, action), 1, self.depth)
            if score > max_score:
                max_score, ideal_action = score, action
        return ideal_action
        


        


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"   
    
    pacmanPos = currentGameState.getPacmanPosition()
    foodPos_list = currentGameState.getFood().asList()
    
    ghostStates = currentGameState.getGhostStates()
    ghostPos_list = [ghostState.getPosition() for ghostState in ghostStates]

    ScaredTimes_list = [ghostState.scaredTimer for ghostState in ghostStates]
    cur_score = currentGameState.getScore()
 
    dis_closest_food = min([util.manhattanDistance(pacmanPos, foodPos) for foodPos in foodPos_list], default=99999999)
    dis_closest_ghost = min([util.manhattanDistance(pacmanPos, ghostState.getPosition()) for ghostState in ghostStates], default=99999999)


    #reciprocal of distance to food
    food_score = 10/dis_closest_food if dis_closest_food != 0 else 0

    #we gave weight -2 for each ghost
    ghost_score_list = [-2 * manhattanDistance(pacmanPos, ghostPos) for ghostPos in ghostPos_list]
    ghost_score = sum(ghost_score_list)

    #we gave positive weight for each scared ghost
    scared_ghost_score_list = [(ScaredTimes_list[0] - manhattanDistance(pacmanPos, ghostPos)) if ScaredTimes_list[0] > 0 else 0 for ghostPos in ghostPos_list]
    scared_ghost_score = sum(scared_ghost_score_list)

    #very near ghost is very threatening
    closest_ghost_score = -5 if dis_closest_ghost <= 2 else 0

    
    finalScore =  cur_score + food_score + ghost_score + scared_ghost_score + closest_ghost_score
    return finalScore


# Abbreviation
better = betterEvaluationFunction
