#!python3

import random 
from math import sqrt

def compute_min_disk(points):
	point_list = list(set(points))
	random.shuffle(point_list)
	#print(point_list)
	return compute_min_disk_with_points(point_list,[])

def compute_min_disk_with_points(interior_points, boundary_points):
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
			disk = compute_min_disk_with_points(interior_points[:i],boundary_points)
			boundary_points.remove(interior_points[i])
			#print('recurse')
			#print(disk)
			continue
		#three points determine an unique boundary
		if len(boundary_points) >= 2:
			disk = circumcircle(boundary_points[0],boundary_points[1],interior_points[i])
			#print('triple')
			#print(disk)
	return disk

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

	
				
class Node:
	"""Represents the local environment of a single node"""
	
	def __init__(self):
		self.inbox = []
		self.local_samples = []
		self.original_local_samples = []
	
	#def compute_infeasible =  


class Point:	
	"""Represents a point in 2D"""
	
	def __init__(self, x,y):
		self.x = x
		self.y = y
	
	def EuclideanDistance(p1,p2):
		return sqrt((p1.x-p2.x)**2+ (p1.y-p2.y)**2)
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
	print(compute_min_disk([L,I,K,H,J]))