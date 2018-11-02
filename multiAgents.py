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
from game import Actions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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
        numFood = successorGameState.getNumFood()
        if numFood != 0:
          foodDist = min([util.manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()])
        else:
          foodDist = 0
        ghostDist = min([util.manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])


        a = 100   # numFood
        b = 0.05  # foodDist
        c = 1     # ghostDist

        if numFood != 0:
          x1 = 1.0/numFood
          x2 = 1.0/foodDist
        else:
          x1 = 1000000
          x2 = 1000000

        if numFood < 5:
          a = 1000

        if foodDist < ghostDist:
            a *= 500
            b *= 10
            c /= 10

        if ghostDist != 0:
          x3 = 1 - 1/ghostDist
        else:
          x3 = -10000000000000000000

        
        f = a*x1 + b*x2 + c*x3
        return f

def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
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
        depth = self.depth*gameState.getNumAgents();
        def value(gameState, depth, agent):
          if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState), None
          actionsList = gameState.getLegalActions(agent)
          valList = [value(gameState.generateSuccessor(agent, action), depth-1, (agent+1)%gameState.getNumAgents()) for action in actionsList]
          if agent == 0:
            return max(valList), actionsList[valList.index(max(valList))] 
          else:
            return min(valList), actionsList[valList.index(min(valList))] 
        return value(gameState,depth,0)[1]


        #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        depth = self.depth*gameState.getNumAgents();

        def value(gameState, depth, agent, a, b):
          if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState), None
          actionsList = gameState.getLegalActions(agent)
          
          
          if agent == 0:     # Max agent
            v = float("-inf")
            maxIndex = -1
            for i in range(len(actionsList)):
              nextVal = value(gameState.generateSuccessor(agent, actionsList[i]), depth-1, (agent+1)%gameState.getNumAgents(), a, b)[0]
              if v < nextVal:
                v = nextVal
                maxIndex = i
              if v > b:
                return v,actionsList[maxIndex]
              a = max(a,v)
            return v,actionsList[maxIndex]

          else:     # Min agent
            v = float("inf")
            minIndex = -1
            for i in range(len(actionsList)):
              nextVal = value(gameState.generateSuccessor(agent, actionsList[i]), depth-1, (agent+1)%gameState.getNumAgents(), a, b)[0]
              if v > nextVal:
                v = nextVal
                minIndex = i
              if v < a:
                return v,actionsList[minIndex]
              b = min(b,v)
            return v,actionsList[minIndex]

        return value(gameState,depth,0,float("-inf"),float("inf"))[1]




class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        depth = self.depth*gameState.getNumAgents();
        def value(gameState, depth, agent):
          if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState), None
          actionsList = gameState.getLegalActions(agent)
          valList = [value(gameState.generateSuccessor(agent, action), depth-1, (agent+1)%gameState.getNumAgents())[0] for action in actionsList]
          if agent == 0:
            maxVal = valList[0]
            maxIndices = [0]
            if len(valList) == 1:
              return maxVal, actionsList[0]
            for i in range(1, len(valList)):
              if valList[i] > maxVal:
                maxVal = valList[i]
                maxIndices = [i]
              elif valList[i] == maxVal:
                maxIndices.append(i)
              else:
                pass
            return maxVal, actionsList[random.choice(maxIndices)] 
          else:
            return sum(valList)/float(len(valList)), actionsList[valList.index(min(valList))] 
        return value(gameState,depth,0)[1]





def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isLose():
      return -100000000000
    elif currentGameState.isWin():
      return 100000000000

    score = currentGameState.getScore()
    pacPos = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    numFood = currentGameState.getNumFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    
    #foodDist = max([util.manhattanDistance(pacPos, foodPos) for foodPos in foodGrid.asList()])
    foodDist = min([util.manhattanDistance(pacPos, foodPos) for foodPos in foodGrid.asList()])
    
    mazeDist = lambda curr: mazeDistance(curr, (int(pacPos[0]),int(pacPos[1])), currentGameState);
    #foodDist = min([mazeDist((int(pos[0]),int(pos[1]))) for pos in foodGrid.asList()])
    ghostDist = min([mazeDist( (int(ghost.getPosition()[0]), int(ghost.getPosition()[1])) ) for ghost in ghostStates])

    x1 = score
    x2 = numFood
    x3 = foodDist
    x4 = ghostDist

    a0 = 1    # score
    b0 = -10   # numFood
    c0 = -1    # foodDist

    if ghostDist < 3:
      d0 = -1
    else:
      d0 = 0


    f = a0*x1 + b0*x2 + c0*x3 + d0*x4
    #print(a0*x1, b0*x2, c0*x3, c1, d0*x4, d1)
    return f












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
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

def genericSearch(problem, fringe):
    closed = []
    fringe.push( (problem.getStartState(), [""], 0) )
    while not fringe.isEmpty():
        node = fringe.pop()
        if problem.isGoalState(node[0]):
            return node[1][1:]
        if node[0] not in closed:
            closed.append(node[0])
            for child_state, action, cost in problem.getSuccessors(node[0]):
                child_actions = node[1][:]
                child_actions.append(action)
                fringe.push( (child_state, child_actions, node[2] + cost) )


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    return genericSearch(problem, util.Queue())


class PositionSearchProblem(SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost


def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(bfs(prob))




# Abbreviation
better = betterEvaluationFunction
bfs = breadthFirstSearch


