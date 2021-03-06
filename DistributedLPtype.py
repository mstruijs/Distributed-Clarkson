#!python3

import random, math, time,logging
import matplotlib.pyplot as plt
from distributedLPtypeAlgorithms import LPTypeNetwork

logging.basicConfig(filename="lastrun.log",level=logging.DEBUG,format='%(message)s')

def compute_min_disk(points):
	"""Random incremental algo for min enclosing disk. O(n) on expectation. NB: assumes general position"""
	point_list = list(set(points))
	if len(point_list)==0:
		return ((Point(0,0),0),[])
	if len(point_list)==1:
		return ((point_list[0],0),[point_list[0]])
	random.shuffle(point_list)
	#print(point_list)
	return compute_min_disk_with_points(point_list,[])

def compute_min_disk_with_points(interior_points, boundary_points):
	"""Main recursive function for min enclosing disk gives tuple (disk,basis)"""
	r = len(boundary_points)
	if r==0:	
		p0= interior_points[0]
		p1= interior_points[1]
	elif r==1:
		p0= interior_points[0]
		p1= boundary_points[0]
	elif r==2:
		p0= boundary_points[0]
		p1= boundary_points[1]
	disk = (Point((p0.x+p1.x)/2, (p0.y+p1.y)/2), Point.EuclideanDistance(p0,p1)/2)
	basis = [p0,p1]
	#print('init')
	#print(disk)
	
	for i in range(2-len(boundary_points),len(interior_points)):
		if Point.EuclideanDistance(interior_points[i],disk[0]) <= disk[1]:
			#point is in disk, it suffices
			continue
		#point is not in disk, move to the boundary
		if len(boundary_points) < 2:
			#boundary isn't uniquely determined
			boundary_points.append(interior_points[i])
			(disk,basis) = compute_min_disk_with_points(interior_points[:i],boundary_points)
			boundary_points.remove(interior_points[i])
			#print('recurse')
			#print(disk)
			continue
		#three points determine an unique boundary
		if len(boundary_points) >= 2:
			disk = circumcircle(boundary_points[0],boundary_points[1],interior_points[i])
			basis = (boundary_points[0],boundary_points[1],interior_points[i])
			#print('triple')
			#print(disk)
	return (disk,basis)

def circumcircle(A,B,C):
	#translate A to origin to simplify calculations
	Bt = Point(B.x-A.x,B.y-A.y)
	Ct = Point(C.x-A.x,C.y-A.y)
	D = 2*(Bt.x * Ct.y - Bt.y * Ct.x)
	#compute translated center
	tcenter = Point((Ct.y * (Bt.x**2 + Bt.y**2) - Bt.y * (Ct.x**2 + Ct.y**2))/D,
		(Bt.x * (Ct.x**2 + Ct.y**2) - Ct.x * (Bt.x**2 + Bt.y**2))/D
		)
	#translate center back
	center = Point(tcenter.x + A.x,tcenter.y + A.y)
	return (center, Point.EuclideanDistance(A,center))

def check_disk_cover(disk, points):
	"""Determine whether a disk covers all points"""
	for p in points:
		if Point.EuclideanDistance(p,disk[0]) - disk[1] > 1E-4:
			return False
	return True
			
class Node:
	"""Represents the local environment of a single node"""
	
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
		
class Controller:
	"""Contains variables and knowledge global to all nodes"""
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
					
class Point:	
	"""Represents a point in 2D"""
	
	def __init__(self, x,y):
		self.x = x
		self.y = y
	
	def EuclideanDistance(p1,p2):
		return math.sqrt((p1.x-p2.x)**2+ (p1.y-p2.y)**2)
	#def add(self, p):
	#	return Point()
	
	def __repr__(self):
		return ("[" + str(self.x) + "," + str(self.y) + "]")

"""	Too complicated, just use lists.	
#implement multiset as a list to be able to sample from it.		
class Multiset:
	""""""
	def __init__(self,set=[]):
		data = []
		length = 0
		for i in set:
			data.apppend((i,1))
			length+=1
	
	def add(self,a):
		for (x,count) in data:
			if x==a:
"""				
	
	

def triple_disk(n,width):
	"""Returns a dataset where 3 points define a disk, with all remaining points uniformly random distributed in the interior of that disk"""
	res = [Point(width*math.cos(2*math.pi*i/3),width*math.sin(2*math.pi*i/3)) for i in range(3)]
	disk = (Point(0,0),width)
	for i in range(n-3):
		while(True):
			x = random.uniform(-width,width)
			y = random.uniform(-width,width)
			if Point.EuclideanDistance(Point(x,y),disk[0]) - width < 10E-4:
				res.append(Point(x,y))
				break
	return res

def duo_disk(n,width):
	"""Returns a dataset where 2 points define a disk, with all remaining points uniformly random distributed in the interior of that disk"""
	res = [Point(-width,0),Point(width,0)]
	disk = (Point(0,0),width)
	#print(disk)
	for i in range(n-3):
		while(True):
			x = random.uniform(-width,width)
			y = random.uniform(-width,width)
			if Point.EuclideanDistance(Point(x,y),disk[0]) - width < 10E-4:
				res.append(Point(x,y))
				break
	return res
	
def triangle(n,width):
	"""Returns a dataset where a triangle is defined by 3 points and the remaining points are distributed uniformly at random on the interior of the triangle."""
	res = [Point(width*math.cos(2*math.pi*i/3),width*math.sin(2*math.pi*i/3)) for i in range(3)]
	for i in range(n-3):
		v = [random.uniform(0,1-1E-4) for j in range(2)]
		v.extend([0,1])
		v.sort()
		weights = [v[j+1]-v[j] for j in range(3)]
		x = sum([res[j].x*weights[j] for j in range(3)])
		y = sum([res[j].y*weights[j] for j in range(3)])
		res.append(Point(x,y))
	return res

def perturbed_regular_hull(n,width):
	"""Returns a dataset where all points are randomly perturbed from the vertices of a regular polygon"""
	res = []
	for i in range(n):
		r = random.uniform(0,width*math.sin(math.pi/n)/2)
		phi = random.uniform(0,2*math.pi)
		x = width*math.cos(2*math.pi*i/n) + r*math.cos(phi)
		y = width*math.sin(2*math.pi*i/n) + r*math.sin(phi)
		res.append(Point(x,y))
	return res		

def display_hull(data,n,n_hull,axes=None):
	"""Plots hulls, should probably be moved to the data-crunching module"""
	if n_hull>0:
		hull_x = [p.x for p in data[:n_hull]]
		hull_x.append(data[0].x)
		hull_y = [p.y for p in data[:n_hull]]
		hull_y.append(data[0].y)
	interior_x = [p.x for p in data[n_hull:]]
	interior_y = [p.y for p in data[n_hull:]]
	
	if axes==None:
		fig = plt.figure()
		base = fig.add_subplot(111)		
	else:
		base = axes 
	base.set_xticks([])
	base.set_yticks([])
	base.axis('equal')
	base.plot(interior_x, interior_y, 'bo')
	if n_hull>0:
		base.plot(hull_x,hull_y, 'r-')
		base.plot(hull_x,hull_y, 'ro')
	return base

def test_run(type,k,datasize,acceleration_func= lambda n : 1,min_k=1):
	"""Run and log tests with various parameters"""
	for test_case in [duo_disk,triple_disk,perturbed_regular_hull,triangle]:
		write_result_line("===="+ test_case.__name__+"====")
		for i in range(min_k,k):
			rounds= []
			for j in range(10):
				n=2**i
				controller= Controller(n,test_case(datasize(n),10*datasize(n)),type,acceleration_factor=int(acceleration_func(n)))
				controller.run()
				rounds.append(controller.rounds)
			logging.info(test_case.__name__ + "; Rounds at 2^"+ str(i) +": " + str(sum(rounds)/10))
			mean = sum(rounds)/len(rounds)
			sample_variance = sum([(x-mean)**2 for x in rounds])/(len(rounds)-1)
			write_result_line("Rounds at 2^"+ str(i) + ". mean: " + str(round(mean,2)) 
				+ ", variance: " + str(round(sample_variance,2)) + ", min: " + str(min(rounds)) + ", max: " + str(max(rounds)) + " data: " + str(rounds) )
	

def write_result_line(line):
	with open("results.txt",'a') as res:
		res.write(line+'\n')

def plot_test_data(n,width):
	"""should probably be moved to the data-crunching module"""
	for (method,x) in [(triple_disk,3), (duo_disk,2),(triangle,3)]:
		display_hull(method(n,width),n,x);plt.show()
	data = perturbed_regular_hull(n,width)
	(disk,basis) = compute_min_disk(data)
	basis_x = [p.x for p in basis]
	basis_y = [p.y for p in basis]
	basis_x.append(basis_x[0])
	basis_y.append(basis_y[0])
	ax = display_hull(data,n,0)
	ax.plot(basis_x,basis_y, 'r-')
	ax.plot(basis_x,basis_y, 'ro')
	plt.show()
		
def plot_test_data_final():
	"""should probably be moved to the data-crunching module"""
	n,width = 2**10,10*2**10
	fig, [[ax1,ax2], [ax3,ax4]] =	plt.subplots(2,2)
	ax1.set_xticks([])
	ax1.set_yticks([])
	ax1.axis('equal')
	display_hull(duo_disk(n,width),n,2,ax1)
	
	ax2.set_xticks([])
	ax2.set_yticks([])
	ax2.axis('equal')
	display_hull(triple_disk(n,width),n,3,ax2)

	ax3.set_xticks([])
	ax3.set_yticks([])
	ax3.axis('equal')
	display_hull(triangle(n,width),n,3,ax3)

	ax4.set_xticks([])
	ax4.set_yticks([])
	ax4.axis('equal')
	display_hull(perturbed_regular_hull(n,width),n,3,ax4)
	
	plt.show()

class EnclosingDiskNetwork(LPTypeNetwork):
	"""
	Computes a minimum enclosing disk in 2D
	"""
	def reportline(self,line):
		"""
		The method used by the controller to provide runtime reports, should be implemented by inheriting class.
		"""
		with open("Enclosing_disk.log", 'a') as res:
			res.write(line+'\n')			

	def compute_f(self,dataset):
		"""
		A function that compute the cost of a particular subset of the LP-type problem in question. 
		"""
		return self.compute_solution(dataset)[1]
	
	def compute_solution(self,dataset):
		"""
		A function that compute the solution of a particular subset of the LP-type problem in question. 
		"""
		return compute_min_disk(dataset)[0]
		
	def compute_basis(self,dataset):
		"""
		A function that computes a basis of a particular subset of the LP-type problem in question.
		"""
		return compute_min_disk(dataset)[1]

	def violating_datapoints(self,solution,dataset):
		"""
		Get all datapoints in dataset that violate the given solution
		"""
		res = []
		for p in dataset:
			if Point.EuclideanDistance(p, solution[0]) - solution[1] > 1E-4:
				res.append(p)
		return res
		
	def is_global_solution(self,solution):
		"""
		Tests whether a solution works globally, can be wise to override for the specific problem.
		"""
		for p in self.dataset:
			if Point.EuclideanDistance(p, solution[0]) - solution[1] > 1E-4:
				return False
		return True
		
	def log_at_end_of_round(self):
		"""
		Optional logging at the end of each round
		"""
		max_solution = (Point(0,0),0)
		for node in self.nodes:
			solution = node.result
			if solution[1] > max_solution[1]:
				max_solution = solution
		uncovered = []
		for p in self.dataset:
			if Point.EuclideanDistance(p, max_solution[0]) - max_solution[1] > 1E-4:
				uncovered.append(p)
		self.reportline("round: " + str(self.rounds) + "; data-size: " + str(self.global_data_count) + "; max_solution: " + str(max_solution))
		self.reportline("uncovered: " + str(len(uncovered)) + "; " + str(uncovered[:100]))

	
if __name__ == "__main__":
	"""
	write_result_line("======================================")
	write_result_line("= low load algo: |H|=O(n)            =")
	write_result_line("======================================")
	test_run("low",15,lambda n: n)
	write_result_line("======================================")
	write_result_line("= high load algo: |H|=O(n)           =")
	write_result_line("======================================")
	test_run("high",15,lambda n: n)
	#"""
	"""
	write_result_line("======================================")
	write_result_line("= high load algo: |H|=O(n^2)         =")
	write_result_line("======================================")
	test_run("high",11,lambda n: n**2)
	#"""
	"""
	write_result_line("======================================")
	write_result_line("= high load algo, acc=1: |H|=O(n)    =")
	write_result_line("======================================")
	test_run("high",15,lambda n: n,lambda n: math.log2(n))
	#"""
	"""
	write_result_line("======================================")
	write_result_line("= low load algo: |H|=O(n)           =")
	write_result_line("======================================")
	test_run("low",20,lambda n: n,min_k=14)	
	#"""
	#plot_test_data(2**6,10*2**6)
	n = 2**11
	dataset = duo_disk(n,n*100)
	network = EnclosingDiskNetwork(n,dataset,3,type=EnclosingDiskNetwork.LOW_LOAD)
	print("===Parallel===")
	network.run()
	controller = Controller(n,dataset,type="low")
	print("===Sequential===")
	controller.run()