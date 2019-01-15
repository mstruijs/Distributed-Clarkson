#!python3

import random, math, time,logging
import matplotlib.pyplot as plt

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

def regular_convex_hull_perturbed(n,n_hull,width):
	res= []
	for i in range(n_hull):
		r = random.uniform(0,width*math.sin(math.pi/n_hull)/2)
		phi = random.uniform(0,2*math.pi)
		x = width*math.cos(2*math.pi*i/n_hull) + r*math.cos(phi)
		y = width*math.sin(2*math.pi*i/n_hull)+r*math.sin(phi)
		res.append(Point(x,y))
	#Pick a random convex combination of the hull points 
	for i in range(n-n_hull):
		weights = [random.random() for j in range(n_hull)]
		total_weight = sum(weights)
		x = sum([res[j].x*weights[j]/total_weight for j in range(n_hull)])
		y = sum([res[j].y*weights[j]/total_weight for j in range(n_hull)])
		res.append(Point(x,y))
	return res	

def regular_convex_hull_perturbed_exp(n,n_hull,width):
	res= []
	for i in range(n_hull):
		r = 0#random.uniform(0,width*math.sin(math.pi/n_hull)/2)
		phi = random.uniform(0,2*math.pi)
		x = width*math.cos(2*math.pi*i/n_hull) + r*math.cos(phi)
		y = width*math.sin(2*math.pi*i/n_hull)+r*math.sin(phi)
		res.append(Point(x,y))
		#res.append(Point(x,y)) = [Point(width*math.cos(2*math.pi*i/n_hull),width*math.sin(2*math.pi*i/n_hull)) for i in range(n_hull)]
	#Pick a random convex combination of the hull points 
	for i in range(n-n_hull):
		v = [random.random() for j in range(n_hull-1)]
		v.extend([0,1])
		v.sort()
		weights = [v[j+1]-v[j] for j in range(n_hull)]
		x = sum([res[j].x*weights[j] for j in range(n_hull)])
		y = sum([res[j].y*weights[j] for j in range(n_hull)])
		res.append(Point(x,y))
	return res	
	
def perturb_convex_hull(n,n_hull,width,original_hull):
	res = original_hull(n,n_hull,width)
	for i in range(n_hull):
		r = random.uniform(0,width*math.sin(math.pi/n_hull)/2)
		phi = random.uniform(0,2*math.pi)
		res[i].x += r*math.cos(phi)
		res[i].y += r*math.sin(phi)
	return res
	
	
def irregular_convex_hull(n,n_hull,min_length,max_length,min_angle,max_angle):
	res = [Point(0,0), Point(random.uniform(min_length,max_length),0)]
	prev_angle = 0
	for i in range(2,n_hull):
		#check whether angle is smaller than the line that closes the polygon
		max_allowed_angle = min(math.atan2(res[i-1].y,res[i-1].x)+math.pi,max_angle)
		#print(max_allowed_angle)
		if prev_angle+min_angle> max_allowed_angle:
			angle = random.uniform(prev_angle,max_allowed_angle)
		else:
			angle = random.uniform(prev_angle+min_angle,max_allowed_angle)
		#check whether the point is positive
		if angle > math.pi:
			max_allowed_length = min(res[i-1].y/math.sin(angle-math.pi),max_length)
		else: 
			max_allowed_length = max_length
		if min_length>max_allowed_length:
			length = random.uniform(max_allowed_length/2,max_allowed_length)
		else:
			length = random.uniform(min_length,max_allowed_length)
		res.append(Point(res[i-1].x+length*math.cos(angle),res[i-1].y+length*math.sin(angle)))		
		prev_angle = angle
		#print("angle",angle)
	#Pick a random convex combination of the hull points 
	for i in range(n-n_hull):
		weights = [random.random() for j in range(n_hull)]
		total_weight = sum(weights)
		x = sum([res[j].x*weights[j]/total_weight for j in range(n_hull)])
		y = sum([res[j].y*weights[j]/total_weight for j in range(n_hull)])
		res.append(Point(x,y))
	return res

def triple_disk_old(n,width):
	res = [Point(width*math.cos(2*math.pi*i/3),width*math.sin(2*math.pi*i/3)) for i in range(3)]
	disk = (Point(0,0),width)
	#print(disk)
	for i in range(n-3):
		r = disk[1]*random.uniform(0,1-1E-4)
		phi = random.uniform(0,2*math.pi)
		res.append(Point(disk[0].x + r*math.cos(phi), disk[0].y + r*math.sin(phi)))
	#print(compute_min_disk(res))
	return res

def triple_disk(n,width):
	res = [Point(width*math.cos(2*math.pi*i/3),width*math.sin(2*math.pi*i/3)) for i in range(3)]
	disk = (Point(0,0),width)
	#print(disk)
	for i in range(n-3):
		while(True):
			x = random.uniform(-width,width)
			y = random.uniform(-width,width)
			if Point.EuclideanDistance(Point(x,y),disk[0]) - width < 10E-4:
				res.append(Point(x,y))
				break
			#print(compute_min_disk(res))
	return res

def duo_disk(n,width):
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
			#print(compute_min_disk(res))
	return res
	
	
def duo_disk_old(n,width):
	res = [Point(-width,0),Point(width,0)]
	disk = (Point(0,0),width)
	for i in range(n-3):
		r = disk[1]*random.uniform(0,1-1E-4)
		phi = random.uniform(0,2*math.pi)
		res.append(Point(disk[0].x + r*math.cos(phi), disk[0].y + r*math.sin(phi)))
	#print(compute_min_disk(res))
	return res

def triangle(n,width):
	res = [Point(width*math.cos(2*math.pi*i/3),width*math.sin(2*math.pi*i/3)) for i in range(3)]
	for i in range(n-3):
		v = [random.uniform(0,1-1E-4) for j in range(2)]
		v.extend([0,1])
		v.sort()
		weights = [v[j+1]-v[j] for j in range(3)]
		x = sum([res[j].x*weights[j] for j in range(3)])
		y = sum([res[j].y*weights[j] for j in range(3)])
		res.append(Point(x,y))
	#print(compute_min_disk(res))
	return res

def perturbed_regular_hull(n,width):
	res = []
	for i in range(n):
		r = random.uniform(0,width*math.sin(math.pi/n)/2)
		phi = random.uniform(0,2*math.pi)
		x = width*math.cos(2*math.pi*i/n) + r*math.cos(phi)
		y = width*math.sin(2*math.pi*i/n) + r*math.sin(phi)
		res.append(Point(x,y))
	#print(compute_min_disk(res))
	return res

	
def convex_hull_tests():
	n = 2**13
	for hull_size in [3,4,5,8,13,21,int(n/3),n]:
		#test regular hull
		"""
		rounds = 0
		for i in range(10):
			controller = Controller(n,regular_convex_hull(n,hull_size,10000))
			controller.run_low_load()
			rounds+= controller.rounds
		logging.info("Rounds at regular hull with hull size"+ str(round(hull_size,2)) +": " + str(rounds/10))
		#test irregular hull
		rounds = 0
		for i in range(10):
			controller = Controller(n,irregular_convex_hull(n,hull_size,100,10000,0.01,math.pi))
			controller.run_low_load()
			rounds+= controller.rounds
		logging.info("Rounds at irregular hull with hull size"+ str(round(hull_size,2)) +": " + str(rounds/10))
		#test regular hull with interior convex combinations
		rounds = 0
		for i in range(10):
			controller = Controller(n,regular_convex_hull_convex_combinations(n,hull_size,10000))
			controller.run_low_load()
			rounds+= controller.rounds
		logging.info("Rounds at regular hull with convex interior with hull size"+ str(round(hull_size,2)) +": " + str(rounds/10))
		#test perturbed regular hull
		rounds = 0		
		for i in range(10):
			controller = Controller(n,perturb_convex_hull(n,hull_size,10000,regular_convex_hull))
			controller.run_low_load()
			rounds+= controller.rounds
		logging.info("Rounds at perturbed regular hull with hull size"+ str(round(hull_size,2)) +": " + str(rounds/10))
		#"""
		#test perturbed regular hull with interior convex combinations
		rounds = 0
		for i in range(10):
			controller = Controller(n,regular_convex_hull_perturbed(n,hull_size,10000))
			controller.run_low_load()
			rounds+= controller.rounds
		logging.info("Rounds at perturbed regular hull with convex interior with hull size"+ str(round(hull_size,2)) +": " + str(rounds/10))
		

def display_hull(data,n,n_hull,axes=None):
	if n_hull>0:
		hull_x = [p.x for p in data[:n_hull]]
		hull_x.append(data[0].x)
		hull_y = [p.y for p in data[:n_hull]]
		hull_y.append(data[0].y)
	interior_x = [p.x for p in data[n_hull:]]
	interior_y = [p.y for p in data[n_hull:]]
	
	if axes==None:
		#base = plt
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
	#plt.show()

def test_run(type,k,datasize,acceleration_func= lambda n : 1,min_k=1):
	for test_case in [duo_disk]:#,triple_disk,perturbed_regular_hull,triangle]:
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
	
def plot_hulls():	
	n = 2**13
	hull_size = 3
	display_hull(regular_convex_hull(n,hull_size,10000),n,hull_size)
	display_hull(irregular_convex_hull(n,hull_size,100,10000,0.01,math.pi),n,hull_size)
	display_hull(regular_convex_hull_convex_combinations(n,hull_size,10000),n,hull_size)
	display_hull(perturb_convex_hull(n,hull_size,10000,regular_convex_hull),n,hull_size)
	display_hull(regular_convex_hull_perturbed(n,hull_size,10000),n,hull_size)
	display_hull(regular_convex_hull_perturbed_exp(n,hull_size,10000),n,hull_size)

def plot_test_data(n,width):
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
	
if __name__ == "__main__":
	A = Point(1,0)
	B = Point(3,4)
	C = Point(-1,-1)
	F = Point(2,0)
	G = Point(0,2)
	I = Point(8,8)
	J = Point(12,4)
	H = Point(8,2)
	K = Point(7,4)
	L = Point(10,7)
	#print(circumcircle(A,B,C))
	#print(compute_min_disk([A,B,C,F,G]))
	#print(compute_min_disk([L,I,K,H,J]))
	points = [H,I,J,K,L]
	#n = 10000
	#print(grid_points(10,10,5))
	#controller = Controller(n*n,grid_points(n,n,10))
	#controller = Controller(n,uniform_random_disk(n,10*n**2))
	#controller = Controller(n,small_hull(n,10*n**2))	
	"""
	k=15
	for i in range(k):
		rounds= 0
		for j in range(10):
			n=2**i
			controller= Controller(n,small_hull(n,10*n))
			controller.run_low_load()
			rounds+= controller.rounds
		logging.info("Rounds at 2^"+ str(i) +": " + str(rounds/10))
	"""
	#print(irregular_convex_hull(20,6,2,10,math.pi/4,2*math.pi))
	#convex_hull_tests()
	#plot_hulls()
	#n=2**13
	#hull_size = 3
	#display_hull(regular_convex_hull_perturbed_exp(n,hull_size,10000),n,hull_size)
	#print(compute_min_disk())
	#plot_test_data(2**13,10000)
	#n = 2**8
	#width = 1000
	#controller = Controller(n,triangle(n,width))
	#controller.run_high_load()
	#controller.run_low_load()
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
	plot_test_data(2**6,10*2**6)