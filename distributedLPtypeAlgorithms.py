#!python
"""
Represents the algorithms for an abstract LP-type problem.
"""

import time, random, logging
from threading import Thread,Event,Barrier
from queue import Queue
from abc import ABC, abstractmethod

class LPTypeNode:
	def __init__(self,id,controller,waiting_barrier,event):
		self.controller = controller
		#self.inbox = {}
		#self.outbox = {}
		self.inbox = []
		#for label in message_labels:
		#	self.inbox[label] = []
		#	self.outbox[label] = []
		self.local_samples = []
		self.initial_local_samples = []
		self.result = None
		self.waiting_barrier = waiting_barrier
		self.receive_data_event = event
		self.id = id
		
	def add_initial_sample(self,datapoint):
		self.initial_local_samples.append(datapoint)
	
	def push(self,data):
		"""
		Pushes a list of data to random nodes (via the controller)
		"""
		for p in data:
			self.controller.push_queue.put( (self.id,p) )
		#print(self.id,"entering barrier A")
		self.waiting_barrier.wait()
		
	
	def pull(self):
		"""
		Pulls a list of data from random nodes (via the controller)
		"""
		self.controller.messages_processed_event.wait()
		return self.inbox
	
	def end_round(self):
		"""
		Synchronize with the controller at the end of the round.
		"""
		#print(self.id,"entering barrier B")
		self.waiting_barrier.wait()
		self.controller.round_end_event.wait()

	
	def sample_globally(self):
		return self.controller.get_global_sample(6*self.controller.dimension**2)
		
class LPTypeController(ABC):
	"""
	Generic controller of algorithms for LP-type problems that spawns nodes and manages the global state.
	Specific LP-type problems should inherit this class and apply problem specific functions.
	"""
	LOW_LOAD = 1
	HIGH_LOAD = 2	
	
	def __init__(self,n,datapoints,dimension,type=1,acceleration_factor=1,verification_result=None,initial_distribution_method=None,global_cost = None):
		self.n = n #number of computational nodes
		self.nodes = [] # list of nodes
		self.dataset = datapoints #Dataset to compute on
		self.global_data = list(datapoints) #aggregate of all local data of all nodes, used for sampling, not needed for all algorithms
		self.rounds = 0 #counter for the number of algorithm rounds
		self.dimension = dimension #combinatorial dimension of the problem
		self.result = None #result of computation
		self.start_time = time.time()
		if global_cost == None:
			self.global_cost = self.compute_f(self.dataset) #a result of the computation, possibly provided by some other method to compare against the result of this algorithm
		else:
			self.global_cost = global_cost
		if verification_result == None:
			self.verification_result = self.compute_solution(self.dataset)
		else:	
			self.verification_result = verification_result
		self.global_data_count = 0 #Total number of data objects, useful for algorithms that do not maintain self.global_data
		self.type = type #Type of algorithm to execute
		self.messages_processed_event = Event() #event to signal nodes that messages have been sent by the controller
		self.round_end_event = Event() #event to signal nodes that the round has ended and either the next round begins or thread should end
		self.node_waiting_barrier = Barrier(n+1) #Barrier to notify controller all nodes are waiting
		self.push_queue = Queue() # Queue to receive pushed messages from nodes
		if initial_distribution_method == None:
			self.initial_distribution_method = self.uniform_random
		else:
			self.initial_distribution_method = initial_distribution_method #The method how to perform the original distribution of data to nodes
		#Set labels to identify the message queues
		if self.type == self.LOW_LOAD:
			self.message_labels = ["W"] #The label W represents the datapoints violated by some solution
		if self.type == self.HIGH_LOAD:
			self.message_labels = ["W","Basis"] #The label Basis represents a basis to be shared.
			
		self.acceleration_factor = acceleration_factor #How many times the basis is copied before sending in the high-load algorithm
		self.terminated = False #whether the algorithm has terminated.
	
	@abstractmethod
	def reportline(self,line):
		"""
		The method used by the controller to provide runtime reports, should be implemented by inheriting class.
		"""
		pass
	
	@abstractmethod
	def compute_f(self,dataset):
		"""
		A function that compute the cost of a particular subset of the LP-type problem in question. 
		"""
		pass
	
	@abstractmethod
	def compute_solution(self,dataset):
		"""
		A function that compute the solution of a particular subset of the LP-type problem in question. 
		"""
		pass	

	@abstractmethod
	def compute_basis(self,dataset):
		"""
		A function that computes a basis of a particular subset of the LP-type problem in question.
		"""
		pass
	
	@abstractmethod
	def violating_datapoints(self,solution,dataset):
		"""
		Get all datapoints in dataset that violate the given solution
		"""
		pass
		
	def is_global_solution(self,solution):
		"""
		Tests whether a solution works globally, can be wise to override for specific problem.
		"""
		return self.compute_f(solution) == self.global_cost
		
	def log_partial_solutions(self,solutions):
		"""
		Optional logging of incomplete solutions, does nothing unless overridden
		"""
		pass
	
	def initialise_network(self):

		#construct shared variables and interprocess communication 
		self.node_events = []
		#create n nodes and store them
		for i in range(self.n):
			self.node_events.append(Event())
			self.nodes.append(LPTypeNode(i,self,self.node_waiting_barrier,self.node_events[i]))
		#create some distribution of data to nodes
		distribution = self.initial_distribution_method(self.n,len(self.global_data))
		#distribute data
		for i in range(len(distribution)):
			self.nodes[distribution[i]].add_initial_sample(self.global_data[i])
		#start running nodes in separate processes:
		self.node_threads = []
		for node in self.nodes:
			thread = Thread(target=self.run_node_low_load,args=(node,))
			thread.start()
			self.node_threads.append(thread)
		
	def run_node_low_load(self,node):
		#receive random multiset R of size 3d^2 from controller
		R = node.sample_globally()
		node.result = self.compute_solution(R)
		samples = node.initial_local_samples
		samples.extend(node.local_samples)
		W = self.violating_datapoints(node.result,samples)
		node.push(W)
		received = node.pull()
		node.local_samples.extend(received)
		#Remove samples
		new_local_samples = []
		for h in node.local_samples:
			if random.random() < 1/(1 + 1/(3*self.dimension)):
				new_local_samples.append(h)
		node.local_samples = new_local_samples
		#wait until round ends
		node.end_round()
		#continue to next round
		if not(self.terminated):
			self.run_node_low_load(node)
		#else:
			#print(node.id,"terminated")
	
	def run(self):
		self.initialise_network()
		#self.verified_min_disk = compute_min_disk(self.dataset)[0]
		#terminated = False
		self.reportline("===================================================================================================")
		while(not(self.terminated) and self.rounds <100):
			self.terminated = self.process_round()
			self.round_end_event.set()
		#wait for threads to finish
		for thread in self.node_threads:
			thread.join()
		total_time = time.time() - self.start_time
		timing = "Total time elapsed: " + str(round(total_time,3)) + " s; avg round time: " + str(round(total_time/self.rounds,3)) + " s"
		print(timing)
		self.reportline(timing)
		print("rounds: "+ str(self.rounds))
		print("result: "+ str(self.result))
		print("verify_result: " + str(self.verification_result))

	def process_round(self):
		if self.type == self.LOW_LOAD:
			return self.process_round_low_load()
		if self.type == self.HIGH_LOAD:
			return self.process_round_high_load()

	def process_round_low_load(self):
		"""Process a round. Returns True on termination"""
		n = self.n
		nodes = self.nodes
		self.rounds+=1
		self.round_end_event.clear()
		self.messages_processed_event.clear()
		#No need to process all individual nodes synchronously, just wait
		#for node in nodes:
		#	node.process_round_low_load()
		
		#send samples 
		#for connection in self.node_connections:
		#	connection.send(self.get_global_sample(3*self.dimension**2))
		
		#wait for nodes to finish pushing "W"
		self.node_waiting_barrier.wait()
		#print("reset barrier A")
		self.node_waiting_barrier.reset()
		#TODO process messages, concurrently with sending?
		#node_receiving_messages = []*n
		#divide messages
		while (not(self.push_queue.empty())):
			(send_id, message) = self.push_queue.get()
			#select other node uniformly at random
			receive_id = (send_id + random.randint(1,n-1)) % n
			self.nodes[receive_id].inbox.append(message)
		self.messages_processed_event.set()
		#send them to the nodes
		#for i in range(n):
		#	self.node_connections[i].send(node_receiving_messages[i])
		
		#wait for round end
		self.node_waiting_barrier.wait()
		self.node_waiting_barrier.reset()
		#print("reset barrier B")
		
		#update global data
		self.global_data.clear()
		for node in nodes:
			self.global_data.extend(node.local_samples)
			self.global_data.extend(node.initial_local_samples)
		
		#receive new local solutions
		local_solutions = []
		for node in nodes:
			solution = node.result
			local_solutions.append(solution)
			if self.is_global_solution(solution):
				self.result = solution
				return True
		#perform optional logging
		self.log_partial_solutions(local_solutions)
		return False
	
	def nodes_finished(self):
		for node in self.nodes:
			if not(node.waiting):
				return False
		return True

	def get_global_sample(self,k):
		if k<= len(self.global_data):
			return random.sample(self.global_data, k)
		return list(self.global_data)
		
	def process_round_high_load(self):
		pass
	
	def uniform_deterministic(self,n,m):
		"""Deterministically divides data as uniformly as possible, giving preference to lower ids"""
		return [i % n for i in range(m)]
		
	def uniform_random(self,n,m):
		"""Distributes data uniformly at random over the nodes"""
		return [random.randint(0,n-1) for i in range(m)]		

class Node_sequential:
	"""Represents the local environment of a single node  controlled by Controller_sequential"""
	
	def __init__(self,id,controller,message_labels):
		self.id = id
		self.controller = controller
		self.inbox = {}
		self.outbox = {}
		for label in message_labels:
			self.inbox[label] = []
			self.outbox[label] = []
		self.local_samples = []
		self.initial_local_samples = []
		self.disk = (Point(0,0),0)
	
	def add_initial_sample(self,datapoint):
		self.initial_local_samples.append(datapoint)
	
	def process_round_low_load(self):
		controller = self.controller
		R = controller.get_sample(3*controller.dimension**2)
		self.disk = compute_min_disk(R)[0]
		W = []
		samples = self.local_samples
		samples.extend(self.initial_local_samples)
		for h in samples:
			if Point.EuclideanDistance(h,self.disk[0]) - self.disk[1] > 1E-4:
				W.append(h)
		#Send W to random units the next round
		self.outbox["W"] = W
		#Receive samples from previous round
		self.local_samples.extend(self.inbox["W"])
		self.inbox["W"] = []
		#Remove samples
		#This is effectively the multiset minus, since we _certainly_ keep the initial samples only once
		new_local_samples = []
		for h in self.local_samples:
			if random.random() < 1/(1+ 1/(3*controller.dimension)):
				new_local_samples.append(h)
		self.local_samples = new_local_samples
	
	def process_round_high_load(self):
		if len(self.local_samples)==0:
			self.local_samples = self.initial_local_samples
		controller = self.controller
		(self.disk,basis) = compute_min_disk(self.local_samples)
		self.outbox["Basis"] = [basis] * controller.acceleration_factor
		W = []
		for basis in self.inbox["Basis"]:
			#print(basis)
			disk = compute_min_disk(basis)[0]
			for h in self.local_samples:
				if Point.EuclideanDistance(h, disk[0]) - disk[1] > 1E-4:
					W.append(h)
		self.inbox["Basis"] = []
		self.outbox["W"] = W
		self.local_samples.extend(self.inbox["W"])
		self.inbox["W"] = []
		
class Controller_sequential:
	"""Contains variables and knowledge global to all nodes, does not use actual concurrency"""
	def __init__(self,n,datapoints,type="",acceleration_factor=1):
		self.n = n #number of nodes
		self.nodes = []
		self.dataset = datapoints
		self.global_data = list(datapoints) #assumes datapoints to be unique?
		self.rounds = 0
		self.dimension = 3
		self.result = (Point(0,0),0)
		self.start_time = time.time()
		self.verified_min_disk = (Point(0,0),0)
		self.message_labels = []
		self.global_data_count = 0
		self.type=type
		self.acceleration_factor = acceleration_factor
	
	def run(self):
		if self.type == "high":
			self.run_high_load()
		elif self.type == "low":
			self.run_low_load()
		
	def initialise_network(self,distribution_method,message_labels):
		self.message_labels = message_labels
		#create n nodes and store them
		for i in range(self.n):
			self.nodes.append(Node(i,self,message_labels))
		#create some distribution of data to nodes
		distribution = distribution_method(self.n,len(self.global_data))
		#distribute data
		for i in range(len(distribution)):
			self.nodes[distribution[i]].add_initial_sample(self.global_data[i]) 
	
	def process_round_low_load(self):
		"""Process a round. Returns True on termination"""
		n = self.n
		nodes = self.nodes
		self.rounds+=1
		#process all individual nodes sequentially
		for node in nodes:
			node.process_round_low_load()
		#process messages
		for i in range(n):
			for message in nodes[i].outbox["W"]:
				#select unif random _other_ node
				j = (i + random.randint(1,n-1)) % n
				nodes[j].inbox["W"].append(message)
			nodes[i].outbox["W"] = []
		#update global data
		self.global_data.clear()
		for node in nodes:
			self.global_data.extend(node.local_samples)
			self.global_data.extend(node.initial_local_samples)
		#log
		max_disk = (Point(0,0),0)
		for node in nodes:
			if node.disk[1] > max_disk[1]:
				max_disk = node.disk
		uncovered = []
		true_uncovered_count = 0 #count all points that are also missed by the reference disk
		for point in self.dataset:
			if Point.EuclideanDistance(point,max_disk[0]) - max_disk[1] > 1E-4:
				uncovered.append((point,Point.EuclideanDistance(point,max_disk[0])))
				if not(Point.EuclideanDistance(point,self.verified_min_disk[0]) - self.verified_min_disk[1] > 1E-4):
					true_uncovered_count+=1
		logging.info("round: " + str(self.rounds) + "; data-size: " + str(len(self.global_data)) + "; max_disk: " + str(max_disk))
		logging.info("uncovered: " + str(true_uncovered_count) + "; " + str(uncovered[:100]))
		#check termination
		for node in nodes:
			if check_disk_cover(node.disk, self.dataset):
				self.result = node.disk
				return True
		#Check if the blocking nodes are also blocking on the reference disk
		if true_uncovered_count==0:
			logging.info("NOTE: Disk is found, but does not cover due to rounding error")
			return True
		return False
	
	def process_round_high_load(self):
		n = self.n
		nodes = self.nodes
		self.rounds+=1
		for node in nodes:
			node.process_round_high_load()
		for i in range(n):
			for label in self.message_labels:
				for message in nodes[i].outbox[label]:
					j =  (i + random.randint(1,n-1)) % n
					nodes[j].inbox[label].append(message)
				nodes[i].outbox[label] = []
		#No need to keep track of global data, size is nice for logging
		self.global_data_count = sum([len(node.local_samples) for node in nodes])
		#log 
		max_disk = (Point(0,0),0)
		for node in nodes:
			if node.disk[1] > max_disk[1]:
				max_disk = node.disk
		uncovered = []
		true_uncovered_count = 0 #count all points that are also missed by the reference disk
		for point in self.dataset:
			if Point.EuclideanDistance(point,max_disk[0]) - max_disk[1] > 1E-4:
				uncovered.append((point,Point.EuclideanDistance(point,max_disk[0])))
				if not(Point.EuclideanDistance(point,self.verified_min_disk[0]) - self.verified_min_disk[1] > 1E-4):
					true_uncovered_count+=1
		logging.info("round: " + str(self.rounds) + "; data-size: " + str(self.global_data_count) + "; max_disk: " + str(max_disk))
		logging.info("uncovered: " + str(true_uncovered_count) + "; " + str(uncovered[:100]))		
		#check termination
		for node in nodes:
			if check_disk_cover(node.disk, self.dataset):
				self.result = node.disk
				return True
		#Check if the blocking nodes are also blocking on the reference disk
		if true_uncovered_count==0:
			logging.info("NOTE: Disk is found, but does not cover due to rounding error")
			return True
		return False

		
	def run_low_load(self):
		self.initialise_network(Controller.uniform_random,["W"])
		self.verified_min_disk = compute_min_disk(self.dataset)[0]
		terminated = False
		logging.info("===================================================================================================")
		while(not(terminated) and self.rounds <100):
			terminated = self.process_round_low_load()
		total_time = time.time() - self.start_time
		timing = "Total time elapsed: " + str(round(total_time,3)) + " s; avg round time: " + str(round(total_time/self.rounds,3)) + " s"
		print(timing)
		logging.info(timing)
		print("rounds: "+ str(self.rounds))
		print("result: "+ str(self.result))
		print("verify_result: " + str(self.verified_min_disk))
		
	def run_high_load(self):
		self.initialise_network(Controller.uniform_random,["W","Basis"])
		self.verified_min_disk = compute_min_disk(self.dataset)[0]
		terminated = False
		logging.info("===================================================================================================")
		while(not(terminated) and self.rounds <100):
			terminated = self.process_round_high_load()
		total_time = time.time() - self.start_time
		timing = "Total time elapsed: " + str(round(total_time,3)) + " s; avg round time: " + str(round(total_time/self.rounds,3)) + " s"
		print(timing)
		logging.info(timing)
		print("rounds: "+ str(self.rounds))
		print("result: "+ str(self.result))
		print("verify_result: " + str(self.verified_min_disk))
		
		
	def get_sample(self,k):
		if k<= len(self.global_data):
			return random.sample(self.global_data, k)
		return list(self.global_data)
				
	def uniform_deterministic(n,m):
		"""Deterministically divides data as uniformly as possible, giving preference to lower ids"""
		return [i % n for i in range(m)]
		
	def uniform_random(n,m):
		"""Distributes data uniformly at random over the nodes"""
		return [random.randint(0,n-1) for i in range(m)]
		
if __name__ == "__main__":
	#test with easy LP-type problem
	
	test_dist_filename = "EZ-lp-test.log"
	
	class ShortestEnclosingIntervalController(LPTypeController):
	
		def reportline(self,line):
			"""
			The method used by the controller to provide runtime reports, should be implemented by inheriting class.
			"""
			with open(test_dist_filename, 'a') as res:
				res.write(line+'\n')
				
	
		def compute_f(self,dataset):
			"""
			A function that compute the cost of a particular subset of the LP-type problem in question. 
			"""
			return max(dataset)-min(dataset)
		
		def compute_solution(self,dataset):
			"""
			A function that compute the solution of a particular subset of the LP-type problem in question. 
			"""
			return (min(dataset),max(dataset))
			
		def compute_basis(self,dataset):
			"""
			A function that computes a basis of a particular subset of the LP-type problem in question.
			"""
			return [min(dataset),max(dataset)]

		def violating_datapoints(self,solution,dataset):
			"""
			Get all datapoints in dataset that violate the given solution
			"""
			res = []
			for p in dataset:
				if p < solution[0] or p > solution[1]:
					res.append(p)
			return res
			
		def is_global_solution(self,solution):
			"""
			Tests whether a solution works globally, can be wise to override for specific problem.
			"""
			return solution[0] <= min(self.dataset) and max(self.dataset) <= solution[1]
			
		def log_partial_solutions(self,solutions):
			"""
			Optional logging of incomplete solutions, does nothing unless overridden
			"""
			pass
	n = 10
	dataset = [10*random.random() for i in range(100*n)]
	controller = ShortestEnclosingIntervalController(n,dataset,2)
	controller.run()
	print("rounds"  controller.rounds())