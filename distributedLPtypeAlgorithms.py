#!python
"""
Represents the algorithms for an abstract LP-type problem.
"""

import time, random
from threading import Thread,Barrier
from queue import Queue
from abc import ABC, abstractmethod

class LPTypeNode:
	def __init__(self,id,network):
		self.network = network #network the node belongs to
		self.inbox = {} # inbox of node, used to receive pushed messages from other nodes
		for message_type in self.network.message_types:
			self.inbox[message_type] = []
		for pull_type in self.network.pull_types:
			self.inbox[pull_type] = []
		self.local_samples = [] # local samples of this node, excluding intial samples
		self.initial_local_samples = [] # initial samples of node, these will never be removed
		self.result = None # current result local to node
		self.node_barrier = self.network.node_barrier
		self.id = id #node id, used to determine message origin (so a node cannot send a message to itself)
		self.pull_phase = True # node is in "pull phase" (used only for LOWEST_LOAD algorithm)
		
	def add_initial_sample(self,datapoint):
		self.initial_local_samples.append(datapoint)
		self.pull_phase = False # node has an initial sample, so is not in pull phase

	def push_and_receive(self,data,message_type):
		"""
		Pushes a list of data to random nodes and receieves pushed data from other nodes
		"""
		self.push(data,message_type)
		return self.push_receive(message_type)

	def push(self,data,message_type):
		"""
		Pushes a list of data to random nodes (via the controller). 
		"""
		for message in data:
			self.network.push_message((self.id, message), message_type)
	
	def push_receive(self,message_type):
		"""
		Receives pushed data from random nodes of the given message type
		"""
		res = list(self.inbox[message_type])
		self.inbox[message_type].clear()
		return res
	
	def request_pull(self, pull_type):
		"""
		Send out a pull message from the network and receive replies to previously send pulls
		"""
		self.network.pull_message(self.id, pull_type)
		res = list(self.inbox[pull_type])
		self.inbox[pull_type].clear()
		return res
	
	def end_round(self):
		"""
		Synchronize the nodes and run cleanup at the end of the round.
		"""
		self.control_action(self.network.end_of_round)
	
	def control_action(self,action):
		"""
		Perform action by a single node while all other nodes wait, 
		so the action can safely modify global state
		"""
		barrier_id = self.node_barrier.wait()
		if barrier_id == 0:
			action()
		self.node_barrier.wait()
	
	def sample_globally(self):
		return self.network.get_global_sample(6*self.network.dimension**2)
		
class LPTypeNetwork(ABC):
	"""
	Constructs a network of LPTypeNodes to run distribted algorithms for LP-type problems that.
	This class constructs the nodes and manages the global state.
	Specific LP-type problems should inherit this class and apply problem specific functions.
	"""
	LOW_LOAD = 1
	HIGH_LOAD = 2	
	LOWEST_LOAD = 3
	
	def __init__(self,n,datapoints,dimension,type=1,acceleration_factor=1,verification_result=None,initial_distribution_method=None,global_cost = None,threaded=False):
		self.n = n #number of computational nodes
		self.nodes = [] # list of nodes
		self.dataset = datapoints #Dataset to compute on
		self.global_data = list(datapoints) #aggregate of all local data of all nodes, used for sampling, not needed for all algorithms
		self.rounds = 0 #counter for the number of algorithm rounds
		self.dimension = dimension #combinatorial dimension of the problem
		self.result = None #result of computation
		self.start_time = time.time()
		self.threaded = threaded #Whether the nodes should run on individual threads or not
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
		self.node_barrier = Barrier(n) #Barrier to indicate that all nodes are waiting
		self.push_queues = {} # Contains queues to receive pushed messages from nodes with the given message type
		self.pull_queues = {} # Contains queues to receive requests for pulls for data of the given message type. Each pull queue should have a corresponding push inbox
		if initial_distribution_method == None:
			self.initial_distribution_method = self.uniform_random
		else:
			self.initial_distribution_method = initial_distribution_method #The method how to perform the original distribution of data to nodes
		self.acceleration_factor = acceleration_factor #How many times the basis is copied before sending in the high-load algorithm
		self.terminated = False #whether the algorithm has terminated.
		self.message_types = ["W"] #datapoints that violate local solutions (distributed by other nodes)
		self.pull_types = [] #specific messages for pulls
		if self.type == self.LOWEST_LOAD:
			self.pull_types.append("Pull initial element") #A request to send some of the initial data to another node.
			#self.message_types.append("Pull initial element") #A response to the previous request
			self.message_types.append("Push initial element") #An datapoint that should be added to the initial data.
		if self.type == self.HIGH_LOAD:
			self.message_types.append("Basis") #Sets that form a basis for local solutions
		for message_type in self.message_types:
			self.push_queues[message_type] = Queue() 
		for pull_type in self.pull_types:
			self.pull_queues[pull_type] = Queue()
		

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
		A function that computes a basis of a particular subset of the LP-type problem in question. Note that the dataset may be empty
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
		
	def log_at_end_of_round(self,solutions):
		"""
		Optional logging of incomplete solutions, does nothing unless overridden
		"""
		pass
	
	def initialise_network(self):
		#create n nodes and store them
		for i in range(self.n):
			self.nodes.append(LPTypeNode(i,self))
		#create some distribution of data to nodes
		distribution = self.initial_distribution_method(self.n,len(self.global_data))
		#distribute data
		for i in range(len(distribution)):
			self.nodes[distribution[i]].add_initial_sample(self.global_data[i])
		if self.threaded:
			#start running nodes in separate processes:
			self.node_threads = []
			for node in self.nodes:
				thread = Thread(target=self.run_round_threaded,args=(node,))
				thread.start()
				self.node_threads.append(thread)
		
	def low_load_node_round(self,node):
		#ensure samples are in state before pulling
		if self.type == self.LOWEST_LOAD and node.pull_phase:
			h = node.request_pull("Pull initial element")
			if len(h) > 0:
				node.push(h, "Push initial element")
				node.pull_phase = False
		else:
			#sample random multiset R of size 3d^2 from network
			R = node.sample_globally()
			node.result = self.compute_solution(R)

			samples = node.initial_local_samples
			samples.extend(node.local_samples)
			W = self.violating_datapoints(node.result,samples)
			node.push(W,"W")
		if self.type == self.LOWEST_LOAD:
			received_initial = node.push_receive("Push initial element")
			node.initial_local_samples.extend(received_initial)
		received = node.push_receive("W")
		node.local_samples.extend(received)
		#Remove samples
		new_local_samples = []
		for h in node.local_samples:
			if random.random() < 1/(1 + 1/(3*self.dimension)):
				new_local_samples.append(h)
		node.local_samples = new_local_samples
	
	def run_round_threaded(self,node):		
		#perform round
		if self.type == self.LOW_LOAD or self.type == self.LOWEST_LOAD:
			self.low_load_node_round(node)
		elif self.type == self.HIGH_LOAD:
			self.high_load_node_round(node)
		#wait until all nodes have finished their round
		node.end_round()
		#continue to next round if not terminated
		if not(self.terminated):
			self.run_round_threaded(node)
		#else:
			#print(node.id,"terminated")
	
	def run_round_sequential(self):
		#perform round
		for node in self.nodes:
			if self.type == self.LOW_LOAD or self.type == self.LOWEST_LOAD:
				self.low_load_node_round(node)
			elif self.type == self.HIGH_LOAD:
				self.high_load_node_round(node)
		#finish round
		self.end_of_round()
		return self.terminated
	
	def high_load_node_round(self,node):
		if len(node.local_samples)==0:
			node.local_samples = node.initial_local_samples
		basis = self.compute_basis(node.local_samples)
		#push and receive basis
		received_basis = node.push_and_receive([basis] * self.acceleration_factor, "Basis")
		#push local data that violates received basis
		for basis in received_basis:
			solution = self.compute_solution(basis)
			W = self.violating_datapoints(solution,node.local_samples)
			node.push(W, "W")
		node.local_samples.extend(node.push_receive("W"))
		node.result = self.compute_solution(node.local_samples)
		
	def run(self):
		self.initialise_network()
		self.reportline("===================================================================================================")
		if self.threaded:
			#wait for node threads to finish
			for thread in self.node_threads:
				thread.join()
		else:
			terminated = False
			while not(terminated):
				terminated = self.run_round_sequential()
		total_time = time.time() - self.start_time
		timing = "Total time elapsed: " + str(round(total_time,3)) + " s; avg round time: " + str(round(total_time/self.rounds,3)) + " s"
		self.reportline(timing)
		print(timing)
		print("rounds: "+ str(self.rounds))
		print("result: "+ str(self.result))
		print("verify_result: " + str(self.verification_result))
	
	def distribute_messages(self):
		for message_type in self.message_types:
			push_queue = self.push_queues[message_type]
			while (not(push_queue.empty())):
				(send_id, message) = push_queue.get()
				#select other node uniformly at random
				receive_id = (send_id + random.randint(1,self.n-1)) % self.n
				self.nodes[receive_id].inbox[message_type].append(message)
		for pull_type in self.pull_types:
			pull_queue = self.pull_queues[pull_type]
			while (not(pull_queue.empty())):
				request_id = pull_queue.get()
				#select other node uniformly at random
				receive_id = (request_id + random.randint(1,self.n-1)) % self.n
				#NB: not really abstracted, but is fine for current algorithm
				samples = self.nodes[receive_id].initial_local_samples
				if len(samples) > 0:
					self.nodes[request_id].inbox[pull_type].append(random.choice(samples))
			
	def end_of_round(self):
		self.rounds+=1
		#update global data
		if self.type == self.LOW_LOAD or self.type == self.LOWEST_LOAD:
			self.global_data.clear()
			for node in self.nodes:
				self.global_data.extend(node.local_samples)
				self.global_data.extend(node.initial_local_samples)
			self.global_data_count = len(self.global_data)
		elif self.type == self.HIGH_LOAD:
			self.global_data_count = sum([len(node.local_samples) for node in self.nodes])
		#check local solutions
		global_solution_found = False
		local_solutions=[]
		for node in self.nodes:
			solution = node.result
			if solution == None:
				continue
			local_solutions.append(solution)
			if self.is_global_solution(solution):
				self.result = solution
				global_solution_found = True
				break
		#distribute messages
		self.distribute_messages()
		#perform optional logging
		self.log_at_end_of_round()
		#terminate or continue		
		if global_solution_found or self.rounds>=100:
			self.terminated = True
	
	def push_message(self,message,message_type):
		"""Puts a message in the queue of the appropriate message_type, to be randomly distributed in the next round"""
		self.push_queues[message_type].put(message)
	
	def pull_message(self,request_id,pull_type):
		"""Puts a pull request from the node with id request_id in the queue of the appropriate message_type, to be fullfilled by a random node in the next round"""
		self.pull_queues[pull_type].put(request_id)
		
	def get_global_sample(self,k):
		if k<= len(self.global_data):
			return random.sample(self.global_data, k)
		return list(self.global_data)
			
	def uniform_deterministic(self,n,m):
		"""Deterministically divides data as uniformly as possible, giving preference to lower ids"""
		return [i % n for i in range(m)]
		
	def uniform_random(self,n,m):
		"""Distributes data uniformly at random over the nodes"""
		return [random.randint(0,n-1) for i in range(m)]		
		
if __name__ == "__main__":
	#test with easy LP-type problem
	
	test_dist_filename = "EZ-lp-test.log"
	
	class ShortestEnclosingIntervalNetwork(LPTypeNetwork):
	
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
			if len(dataset) == 0:
				return (0,0)
			return (min(dataset),max(dataset))
			
		def compute_basis(self,dataset):
			"""
			A function that computes a basis of a particular subset of the LP-type problem in question.
			"""
			if len(dataset) == 0:
				return (0,0)
			return (min(dataset),max(dataset))

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
			Tests whether a solution works globally, can be wise to override for the specific problem.
			"""
			return len(solution) > 0 and solution[0] <= min(self.dataset) and max(self.dataset) <= solution[1]
			
		def log_at_end_of_round(self):
			"""
			Optional logging at the end of each round
			"""
			max_solution = (0,0)
			for node in self.nodes:
				solution = node.result
				if solution!=None and solution[1]-solution[0] > max_solution[1] - max_solution[0]:
					max_solution = solution
			uncovered = []
			for point in self.dataset:
				if point < max_solution[0] or point > max_solution[1]:
					uncovered.append(point)
			self.reportline("round: " + str(self.rounds) + "; data-size: " + str(self.global_data_count) + "; max_solution: " + str(max_solution))
			self.reportline("uncovered: " + str(len(uncovered)) + "; " + str(uncovered[:100]))
			if self.type == self.LOWEST_LOAD:
				self.reportline("nodes in pull-phase: " + str(sum([1 if n.pull_phase else 0 for n in self.nodes])))
			
	n = 2**11
	dataset = [10*random.random() for i in range(n)]
	dataset.extend([0,10])
	network = ShortestEnclosingIntervalNetwork(n,dataset,2,type=ShortestEnclosingIntervalNetwork.LOW_LOAD,threaded=False)
	network.run()
	network = ShortestEnclosingIntervalNetwork(n,dataset,2,type=ShortestEnclosingIntervalNetwork.LOW_LOAD,threaded=True)
	network.run()
	#print("rounds",network.rounds)