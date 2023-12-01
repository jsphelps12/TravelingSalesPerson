#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))


import time
import numpy as np
from TSPClasses import *
import heapq
import itertools


class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution,
		time spent to find solution, number of permutations tried during search, the
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

	def greedy( self, time_allowance=60.0):
		# greedy algorithm to find an initial BSSF
		cities = self._scenario.getCities()
		cost, matrix, lowestEdges = self.createRCMatrix(cities)
		greedy_matrix = matrix.copy()
		greedy_cost = cost
		path = []
		results = {}
		# starting our greedy search from node 0
		initialNode = 0
		cities = self._scenario.getCities()
		ncities = len(cities)
		path.append(initialNode)
		# do while we don't have a complete route
		start_time = time.time()
		foundTour = False
		while len(path) != len(cities):
			# if time.time()-start_time >=60.0:
			# 	return None

			minVal = min(greedy_matrix[path[-1],:])
			#if we can't complete the tour we reset
			if minVal == np.inf:
				path.clear()
				initialNode+=1
				path.append(initialNode)
				greedy_matrix = matrix.copy()
				greedy_cost = cost
				continue
			#find column of minimum value and infinity out appropriate rows and columns
			col_index = np.argmin(greedy_matrix[path[-1]])
			greedy_matrix[path[-1],:] = np.inf
			greedy_matrix[:,col_index] = np.inf
			greedy_matrix[col_index,path[-1]] = np.inf
			#append column value to the path as the next city being visited
			path.append(col_index)
			greedy_cost += minVal
			if len(path) == len(greedy_matrix):
				#if we have a complete tour, but no return, we should reset
				if greedy_matrix[path[-1],initialNode] == np.inf:
					path.clear()
					initialNode += 1
					path.append(initialNode)
					greedy_matrix = matrix.copy()
					greedy_cost = cost
					continue
				foundTour = True
				greedy_cost += greedy_matrix[path[-1], initialNode]
		# return as initial bssf
		end_time = time.time()
		bssf = BSSF(greedy_cost, path)
		citiesPath = []
		for i in range(len(bssf.route)):
			citiesPath.append(cities[bssf.route[i]])
		bandbSolution = TSPSolution(citiesPath)
		results ={}
		results['cost'] = bandbSolution.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['soln'] = bandbSolution
		results['count'] = 1 if foundTour else 0
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints:
		max queue size, total number of states created, and number of pruned states.</returns>
	'''

	def branchAndBound( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		# pruned states
		results['pruned'] = 0
		# total states
		results['total'] = 0
		# max queue size
		results['max'] = 0
		# number of solutions found
		results['count'] = 0
		# use priority queue to keep track of next subproblem to search
		priority_queue = []
		# dictionary mapping priority hashing to sub problem
		priority_dict = {}
		# get initial reduction cost, reduced matrix, and list of lowestEdges
		cost, matrix, lowestEdges = self.createRCMatrix(cities)
		# call greedy to get initial bssf
		greedy_bssf = self.greedy()
		if greedy_bssf['cost'] != np.inf:
			foundTour = True
			temp_path = []
			for i in greedy_bssf['soln'].route:
				temp_path.append(i._index)
			bssf = BSSF(greedy_bssf['cost'], temp_path)
		else:
			bssf = BSSF(np.inf, None)
		start_time = time.time()
		timed_out = False
		# do the initial work of adding node 0 to path, making the first subproblem, and adding it to the queue
		initial_node = 0
		cur_path = [initial_node]
		initial_problem = SubProblem(cost, matrix, cur_path)
		results['total'] += 1
		self.addToQueue(initial_problem,priority_queue, priority_dict)
		# while the tour isn't complete we need to go through all the subproblems
		while len(priority_queue) != 0:
			# to start iteration we sort the queue and pop the highest priority item
			priority_queue.sort(reverse=True)
			cur_problem_score = priority_queue.pop()
			cur_problem = priority_dict[cur_problem_score]
			for j in range(ncities):
				# check to see if we time out
				if time.time() - start_time > 60.0:
					timed_out = True
					break
				# create new child problem of the subproblem we have popped from the queue
				problemChild = self.expandSub(cur_problem,j, results)
				# if problemChild is null, we continue
				if problemChild == None:
					continue
				results['total'] += 1
				# check if we have a complete tour
				if len(problemChild.path) == ncities:
					#  check to see if we have a way back
					if problemChild.matrix[problemChild.path[-1],problemChild.path[0]] != np.inf:
						foundTour = True
						problemChild.cost += problemChild.matrix[problemChild.path[-1], problemChild.path[0]]
						# check to see if we will be updating the bssf
						if problemChild.cost < bssf.cost:
							results['count'] += 1
							bssf.cost = problemChild.cost
							bssf.route = problemChild.path

							continue
						else:
							results['pruned'] += 1
					else:
						results['pruned'] += 1
				# is the cost of the child problem greater than bssf? If so, we prune
				elif problemChild.cost >= bssf.cost:
					results['pruned'] += 1
				# not complete and lower than bssf, we add this child to the queue
				else:
					self.addToQueue(problemChild,priority_queue,priority_dict)
					if len(priority_queue) > results['max']:
						results['max'] = len(priority_queue)
			if timed_out:
				break
		# convert the path into a list of cities
		citiesPath = []
		for i in range(len(bssf.route)):
			citiesPath.append(cities[bssf.route[i]])
		bandbSolution = TSPSolution(citiesPath)

		end_time = time.time()

		results['cost'] = bandbSolution.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['soln'] = bandbSolution
		return results

	def expandSub(self, subProblem,col, results):
		# if the city is explored and the point in the grid is infinity, we return none
		if subProblem.matrix[subProblem.path[-1],col] == np.inf:
			return None
		# copy matrix and infinity out appropriate rows, columns, and reverse box
		matrix_copy = subProblem.matrix.copy()
		matrix_copy[subProblem.path[-1]] = np.inf
		matrix_copy[:,col] = np.inf
		matrix_copy[col, subProblem.path[-1]] = np.inf
		reductionCost = 0
		# reduce the matrix and keep track of costs
		for i in range(len(subProblem.matrix[0])):
			rowMin = min(matrix_copy[i,:])
			if rowMin != np.inf:
				matrix_copy[i] -= rowMin
				reductionCost += rowMin
		for j in range(len(subProblem.matrix[0])):
			colMin = min(matrix_copy[:,j])
			if colMin != np.inf:
				matrix_copy[:,j] -= colMin
				reductionCost += colMin
		# calculate new cost, and path, then return
		new_cost = reductionCost + subProblem.cost + subProblem.matrix[subProblem.path[-1],col]
		new_path = subProblem.path.copy()
		new_path.append(col)
		problemChild = SubProblem(new_cost,matrix_copy,new_path)
		return problemChild


	def addToQueue(self, subProblem, priority_queue, priority_dict):
		# add to both queue and dictionary
		hash_value = subProblem.cost/len(subProblem.path)
		priority_queue.append(hash_value)
		priority_dict[hash_value] = subProblem


	#function that creates the initial matrix and then reduces it
	def createRCMatrix(self,cities):
		# initialize matrix with infinity values
		initMatrix = np.full((len(cities), len(cities)), np.inf)
		reductionCost = 0
		lowestEdges = []
		# initialize distances between cities
		for i in range(len(cities)):
			for j in range(len(cities)):
				if i != j:
					initMatrix[i,j] = cities[i].costTo(cities[j])
		# reduce matrix, first rows, and then columns
		for i in range(len(cities)):
			rowMin = min(initMatrix[i,:])
			lowestEdges.append(rowMin)
			if rowMin != 0:
				initMatrix[i] -= rowMin
			reductionCost += rowMin
		for j in range(len(cities)):
			colMin = min(initMatrix[:,j])
			if colMin != 0:
				initMatrix[:,j] -= colMin
			reductionCost += colMin
		lowestEdges.sort()
		# return cost, matrix, and lowest edge from each row before reduction
		return reductionCost, initMatrix, lowestEdges




class BSSF:
	# bssf class holding cost and path
	def __init__(self, score, route):
		self.cost = score
		self.route = route


class SubProblem:
	def __init__(self, cost, matrix, path):
		# subproblem class holding cost, matrix, and path
		self.cost = cost
		self.matrix = matrix
		self.path = path
