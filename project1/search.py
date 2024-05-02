# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        move, stepCost), where 'successor' is a successor to the current
        state, 'move' is the move required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfmoves(self, moves):
        """
         moves: A list of moves to take

        This method returns the total cost of a particular sequence of moves.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    '''
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of moves that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    '''
    fringe=util.Stack()
    visited=set()

    fringe.push((problem.getStartState(),[])) #push((34, 16), []), the list is for path recording
    while not fringe.isEmpty():
        node, path = fringe.pop()
        #print(node)
        if problem.isGoalState(node):
            return path

        if node not in visited:
            visited.add(node)
            for successor, direction, _ in problem.getSuccessors(node):
                newpath = path + [direction]
                fringe.push((successor, newpath))

    return []

    '''
    Start: (34, 16)
    Is the start a goal? False
    Start's successors: [((34, 15), 'South', 1), ((33, 16), 'West', 1)]
    '''

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    fringe=util.Queue()
    visited=set()

    fringe.push((problem.getStartState(),[])) #push((34, 16), []), the list is for path recording
    while not fringe.isEmpty():
        node, path = fringe.pop()
        #print(node)
        if problem.isGoalState(node):
            return path

        if node not in visited:
            visited.add(node)
            for successor, direction, _ in problem.getSuccessors(node):
                newpath = path + [direction]
                fringe.push((successor, newpath))            
    return []

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    fringe=util.PriorityQueue()
    visited=set()

    fringe.push((problem.getStartState(),[],0),0)
    while not fringe.isEmpty():
        node, path, total_cost = fringe.pop()

        if problem.isGoalState(node):
            return path

        if node not in visited:
            visited.add(node)
            for successor, direction, cost in problem.getSuccessors(node):
                newpath=path+[direction]
                new_total_cost=total_cost+cost
                fringe.push((successor, path + [direction], new_total_cost), new_total_cost)
    return []


    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    fringe = util.PriorityQueue()
    visited = set()
    start_state = problem.getStartState()
    initial_h_cost = heuristic(start_state, problem)

    fringe.push((start_state, [], 0), 0 + initial_h_cost)
    while not fringe.isEmpty():
        node, path, g_cost = fringe.pop()

        if problem.isGoalState(node):
            return path

        if node not in visited:
            visited.add(node)
            for successor, direction, cost in problem.getSuccessors(node):
                if successor not in visited:
                    new_g_cost = g_cost + cost
                    h_cost = heuristic(successor, problem)
                    fringe.push((successor, path + [direction], new_g_cost), new_g_cost + h_cost)  # 우선순위는 총 비용
    return []

    
    #searchAgents.nullHeuristic
    #nullHeuristic
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
