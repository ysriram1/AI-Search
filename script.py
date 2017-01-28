# -*- coding: utf-8 -*-

"""
Created on Tue Jan 10 21:58:21 2017

@author: Sriram
"""

import copy
import numpy as np
import time

goalState = [[1,2,3],[8,0,4],[7,6,5]]

easy = [[1, 3, 4], [8, 6, 2], [7, 0, 5]]
medium = [[2,8,1],[0,4,3],[7,6,5]]
hard = [[5,6,7],[4,0,8],[3,2,1]]


###############################################################################
# state: current arrangement of numbers in 8 puzzle (list of lists)
# node: state + information and methods

class node(object):
    
    def __init__(self, state):
        
        self.state = state
        
    def findZeroInState(self):
        '''
        returns the location of 0 of the state
        '''
        
        for out_i, in_lst in enumerate(self.state):
            for in_i, in_val in enumerate(in_lst):
                if in_val == 0: 
                    return out_i, in_i
                    
    def successorMoves(self):
        '''
        returns the possible moves for a given state
        '''
    
        movesDict = {
            # (row, col)
            # blank position : list of possible moves
            (1,1) : [(0,1),(2,1),(1,0),(1,2)], # center
            (0,0) : [(1,0),(0,1)], # top left corner
            (0,2) : [(1,2),(0,1)], # top right corner
            (2,0) : [(1,0),(2,1)], # bottom left corner
            (2,2) : [(1,2),(2,1)], # bottom right corner
            (0,1) : [(0,0),(0,2),(1,1)], # top middle
            (2,1) : [(2,0),(2,2),(1,1)], # bottom middle
            (1,0) : [(2,0),(1,1),(0,0)], # left middle
            (1,2) : [(0,2),(2,2),(1,1)] # right middle
        }
        
        # finding index of blank (0)
        iblank = self.findZeroInState()
                
        return movesDict[iblank]
    
    def createNewNode(self, move, returnValueMoved = False):
        '''
        returns a new node based on a move (input parameter). If returnValueMoved
        is set to True, the value that swapped places with 0 is returned as well.
        '''
        
        # creating a copy of the original node
        newNodeAsLst = copy.deepcopy(self.state)
    
        # find zero
        curr_row, curr_col = self.findZeroInState()
        
        new_row, new_col = move
        
        if returnValueMoved: 
        
            nodeCost = self.state[new_row][new_col] # cost = value being moved
    
        newNodeAsLst[curr_row][curr_col] = self.state[new_row][new_col]
        newNodeAsLst[new_row][new_col]  = 0 # the blank is swapped
    
        if returnValueMoved:
            return node(newNodeAsLst), nodeCost
        
        else:
            return node(newNodeAsLst)
            
    
    def h1Cost(self, goal=goalState):
        '''
        returns the cost associated with a given state, which is the sum of the
        total number of mismatched entries. By default the current state is 
        compared to the goal state.
        '''
        
        goal = np.array(goal)
        node = np.array(self.state)
        
        # we only look at values that are off
        torfMat = goal != node
    
        return sum(sum(torfMat))
        
    def h2Cost(self, goal=goalState):
        '''
        returns the cost associated with a given state as the sum of the
        manhattan distances of mismatched entries of the state compared to their 
        positions in the goal state
        '''
        
        
        goal = np.array(goal)
        node = np.array(self.state)
        
        # we only look at values that are off
        torfMat = goal==node
        falseIdxLst = np.where(torfMat == False)
        
        falseIdxLst = zip(falseIdxLst[0],falseIdxLst[1])
        
        manhattanDist = 0
        
        for falseIdx in falseIdxLst:
                    
            targetValue = node[falseIdx]
            
            goalIdx = np.where(goal == targetValue)
            
            manhattanDist += np.abs(falseIdx[0] - goalIdx[0][0]) + np.abs(falseIdx[1] - goalIdx[1][0])
            
        return manhattanDist

    def h3Cost(self, goal=goalState):
        '''
        returns the cost associated with a given state as the sum of the
        euclidean distances of mismatched entries of the state compared to their 
        positions in the goal state
        '''
        
        goal = np.array(goal)
        node = np.array(self.state)
        
        # we only look at values that are off
        torfMat = goal==node
        falseIdxLst = np.where(torfMat == False)
        
        falseIdxLst = zip(falseIdxLst[0],falseIdxLst[1])
        
        euDist = 0
        
        for falseIdx in falseIdxLst:
                    
            targetValue = node[falseIdx]
            
            goalIdx = np.where(goal == targetValue)
            
            euDist += ((falseIdx[0] - goalIdx[0][0])**2 + (falseIdx[1] - goalIdx[1][0])**2)**0.5
            
        return int(euDist)
        
        
    def isGoal(self, goal=goalState):
        '''
        returns True if state is same as goal state else False
        '''
        return self.state == goal

###############################################################################
class path:
    '''
    collection of methods for creating tree and finding path
    '''  
    # add the tree, printPath, pathFromTree functions in here

    @staticmethod
    def findZeroInState(state):
        '''
        returns the location of 0 in a state        
        '''
        for out_i, in_lst in enumerate(state):
            for in_i, in_val in enumerate(in_lst):
                if in_val == 0: 
    
                    return out_i, in_i
    
    @staticmethod
    def generate(goalNode):
        '''
        returns path, path len, cost of each state, final path cost by iteratively 
        calling parent nodes starting with the goal node. 
        goalNode: The node that corresponds to the final goal state found via search
        '''
        
        path_ = []
        costLst = []
           
        targetNode = goalNode
        
        while targetNode != 'Root':
            
            path_.append(targetNode.state)
            costLst.append(targetNode.totalCost)
            
            targetNode = targetNode.parent
            
        
        path_.reverse() # root first and goal last
        costLst.reverse()        
        
        pathLen, finalPathCost = len(path_), goalNode.totalCost
        
        return path_, pathLen, finalPathCost, costLst
    
    
    @staticmethod    
    def moveDirection(parentState, childState):
        '''
        returns the move direction of 0 based on the change from original state
        (parent state) and new state (child state)
        '''
        
        parent_i, parent_j = path.findZeroInState(parentState)
        child_i, child_j = path.findZeroInState(childState)
        
        iDelta = parent_i - child_i
        jDelta = parent_j - child_j
        
        if iDelta == 1: moveDirection = 'UP'
        if iDelta == -1: moveDirection = 'DOWN'
        if jDelta == 1: moveDirection = 'LEFT'
        if jDelta == -1: moveDirection = 'RIGHT'
        
        return moveDirection    
        
    @staticmethod
    def printState(state):
        '''
        returns a printable format of a state.
        Example: 
        print printState([[1,2,3],[8,0,4],[7,6,5]])
        >>> [1 2 3
             8 0 4
             7 6 5]
        '''
        
        printState = '[' + ','.join(str(x) for x in state[0]) + '\n '\
                    + ','.join(str(x) for x in state[1])\
                      + '\n ' + ','.join(str(x) for x in state[2]) + ']'
        return printState


    
    @staticmethod
    def printPath(path_, costLst):
        '''
        prints entire path: the root to the goal including the move directions
        and the path cost at that node
        '''
        printStatement = ''    
        
        for i, state in enumerate(path_):
            
            if i == 0: printStatement += 'START STATE \n'
            
            printStatement += path.printState(state) + '\n\n\n'
            
            if i == len(path_)-1: 
                printStatement += 'DONE!'
                continue            
            
            printStatement += 'MOVE BLANK ' + path.moveDirection(state, path_[i+1]) + '\n'
            printStatement += 'COST: ' + str(costLst[i+1]-costLst[i]) + '\n'
            printStatement += 'TOTAL COST: ' + str(costLst[i+1]) + '\n'
            
                
        print printStatement
        
###############################################################################
def search(initState, searchAlgo='breathFirst', maxDepth=-1, h=None, printPath=True):
    
    '''
    returns and prints the path and various statistics about it used to find the 
    solution for a given 8-puzzle configuration. 
    
    parms:
    initState: the initial state of the 8-puzzle (nested list)
    searchAlgo: breathFirst, depthFirst, uniformCost, bestFirstCost, AStarCost
    maxDepth: set the maximum depth to traverse the search tree
    h: the heuristic to be used to estimate cost 'h1', 'h2', or 'h3'
        h1: Mismatch, h2: Manhattan, h3: Euclidean
    printPath: if set to True, prints out the path
    '''

    # searchAlgo: breathFirst, depthFirst, uniformCost, bestFirstCost, AStarCost
    # set h=None for all others but bestFirstCost, AStarCost
    # set h to h1,h2, or h3 for AStar. 1: Mismatch, 2: Manhattan, 3: Euclidean

    startTime = time.time()
    
    initNode = node(initState)
    initNode.depth = 0
    initNode.parent = 'Root' # since it is root node
    initNode.g_n = 0
    if h is None: initNode.h_n = 0 # no heuristic used
    if h == 'h1': initNode.h_n = initNode.h1Cost()
    if h == 'h2': initNode.h_n = initNode.h2Cost()
    if h == 'h3': initNode.h_n = initNode.h3Cost()  
    initNode.totalCost = initNode.g_n + initNode.h_n
    
    maxMem = 0 # to store memory usage
    nodeTraverseCount = 0 # to store number of nodes popped

    nodeQ = [initNode] # list to hold the nodes in queue
    visited = [] # list of previously traversed states
    if 'Cost' in searchAlgo:
        costQ = [initNode.totalCost] # list of node costs
    if searchAlgo == 'uniformCost': # to store different goal results for uniformCost search
        goalNodes = []
        goalNodesCost = []
        
    # the search 
    while True:
        
        maxMem = max(maxMem, len(nodeQ))
        
        try:
            if  searchAlgo == 'breathFirst':
                currNode = nodeQ.pop(0) # pop out first node of Queue (FIFO)
            if searchAlgo == 'depthFirst':
                currNode = nodeQ.pop() # pop out last node of Queue (LIFO)
            if 'Cost' in searchAlgo:
                lowestCostIdx = costQ.index(min(costQ))
                costQ.pop(lowestCostIdx) # get index of lowest cost node             
                currNode = nodeQ.pop(lowestCostIdx) # pop out min cost node
                
        except: # if all nodes in queue are traversed (applies to iterative deepening)
            break
        
        if currNode.isGoal(): 
            print 'Solution Found!\n Here is the path:\n'
            goalNode = currNode
            # search ends for all but uniform cost - there may still be a cheaper path
            if searchAlgo != 'uniformCost': break
            goalNodes.append(goalNode)
            goalNodesCost.append(goalNode.totalCost)
            # uniform cost search ends when the following conditions are met
            if goalNode.totalCost < min(costQ): break 
            if len(goalNodesCost) > 0: # if there are multiple paths         
                if min(goalNodesCost) <= min(costQ): break
        
        
        # if max depth reached traverse through the other already added nodes (applies to iterative deepening)
        if  currNode.depth > maxDepth and maxDepth != -1: continue
        
        nextMoves = currNode.successorMoves()
        nodeTraverseCount += 1    
        # add here to visited and not below because node not expanded yet
        if 'Cost' in searchAlgo: visited.append(currNode.state)
        
        for move in nextMoves:
            
            childNode, g_n = currNode.createNewNode(move, returnValueMoved=True) 
            
            # cost should not include g_n for best first
            if searchAlgo == 'bestFirstCost': g_n = 0 

            if childNode.state not in visited:
                
                childNode.parent = currNode 
                childNode.depth = currNode.depth + 1
                childNode.g_n = g_n # the value moved (the cumulation is done in total cost)
                if h is None: childNode.h_n = 0 
                if h == 'h1': childNode.h_n = childNode.h1Cost()
                if h == 'h2': childNode.h_n = childNode.h2Cost()
                if h == 'h3': childNode.h_n = childNode.h3Cost()                
                childNode.totalCost = currNode.totalCost + childNode.g_n + childNode.h_n
                
                nodeQ.append(childNode)
                # only add states to visited for non-cost related algorithms
                if 'Cost' not in searchAlgo: visited.append(childNode.state)
                if 'Cost' in searchAlgo: costQ.append(childNode.totalCost)
    
    # if search ends prior to reaching goal (applies only to iterative deepening)
    if searchAlgo != 'uniformCost':
        if currNode.isGoal():
            path_, pathLen, finalPathCost, pathCostLst = path.generate(goalNode) 
            if printPath: path.printPath(path_,pathCostLst)
            
    else: # to select cheapest goal node for uniform cost search
        minGoalCostIdx = goalNodesCost.index(min(goalNodesCost))
        goalNode = goalNodes.pop(minGoalCostIdx) # pop out min cost node
        path_, pathLen, finalPathCost, pathCostLst = path.generate(goalNode) 
        if printPath: path.printPath(path_,pathCostLst)
        
    endTime = time.time()        
                
    timeTaken = endTime - startTime
    
    if currNode.isGoal():
        return path_, timeTaken, maxMem, nodeTraverseCount, pathLen, \
        finalPathCost, goalNode.depth
    else: # applied only to iterative deepening
        return None, timeTaken, maxMem, nodeTraverseCount, None, None,\
        currNode.depth
        

def ids(initialNode): 
    '''
    returns the solution path and stats related to path using iterative deepening
    search. 
        
    parms:
    initState: the initial state of the 8-puzzle (nested list)
    '''    
    
    totalNodeTraverseCount = 0 # total number of nodes expanded
    totalTime = 0 # time taken to complete traversal   
    depth = 0 # first level   
    maxMem = 0 # maxmimum memory used
    
    while True:
        
        # for each depth level we essentially perform a DFS
    
        print 'Searching Depth of ' + str(depth+1)
        
        path_, timeTaken, maxMem_i, nodeTraverseCount, pathLen, pathCost, \
        maxDepth = search(initialNode, searchAlgo='depthFirst', maxDepth=depth)

        totalNodeTraverseCount += nodeTraverseCount        
        maxMem = max(maxMem, maxMem_i)        
        totalTime += timeTaken
        
        if path_ is not None: break
            
        depth += 1
        
    return path_, totalTime, maxMem, totalNodeTraverseCount, pathLen, pathCost,\
    maxDepth

##############################################################################

# test runs

# easy
search(easy,'breathFirst')
search(easy, 'depthFirst')
ids(easy)
search(easy, 'uniformCost')
search(easy, 'bestFirstCost', h='h1')
search(easy, 'AstarCost', h='h1')
search(easy, 'AstarCost', h='h2')
search(easy, 'AstarCost', h='h3')

# medium
search(medium,'breathFirst')
search(medium, 'depthFirst')
ids(medium)
search(medium, 'uniformCost')
search(medium, 'bestFirstCost', h='h1')
search(medium, 'AstarCost', h='h1')
search(medium, 'AstarCost', h='h2')
search(medium, 'AstarCost', h='h3')

# hard
search(hard,'breathFirst')
search(hard, 'depthFirst')
ids(hard)
search(hard, 'uniformCost')
search(hard, 'bestFirstCost', h='h1')
search(hard, 'AstarCost', h='h1')
search(hard, 'AstarCost', h='h2')
search(hard, 'AstarCost', h='h3')

##############################################################################





    