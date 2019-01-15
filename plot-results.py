#!python3

import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_res(data,labels,start_point=0):
	mpl.rc('text', usetex = True)
	keys = list(data.keys())
	keys.sort()
	keys.reverse()
	for test_case in keys:
		x_val = [i for i in range(start_point+1,len(data[test_case])+1)]
		#y_val2 = [max(rounds) for rounds in data[test_case]]
		y_val = [sum(rounds)/10 for rounds in data[test_case][start_point:]]
		plt.plot(x_val,y_val,labels[test_case],label=str(test_case))
	plt.xlabel(r'$2^i$ nodes')
	plt.ylabel("average rounds until termination")
	plt.legend()
	plt.show()

if __name__ == "__main__":
	data = {}
	data["duo-disk"] = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [2, 1, 1, 1, 1, 1, 1, 1, 2, 1],
                        [1, 1, 7, 1, 1, 1, 4, 1, 1, 5],
                        [3, 1, 2, 1, 4, 1, 2, 2, 2, 2],
                        [1, 2, 3, 4, 1, 4, 5, 1, 2, 3],
                        [9, 3, 8, 2, 1, 5, 5, 3, 8, 8],
                        [8, 4, 6, 3, 8, 6, 10, 8, 5, 1],
						[9, 12, 7, 7, 12, 8, 6, 6, 4, 9],
						[4, 9, 6, 7, 11, 6, 9, 12, 12, 13],
						[10, 12, 13, 12, 10, 12, 10, 12, 11, 13]]
	data["triple-disk"] = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
						  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
						  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
						  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
						  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
						  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
						  [1, 2, 1, 1, 1, 1, 1, 1, 5, 2],
						  [3, 3, 3, 3, 3, 3, 1, 5, 2, 7],
						  [3, 5, 7, 9, 5, 6, 4, 7, 1, 8],
						  [7, 4, 4, 8, 3, 8, 8, 6, 7, 2],
						  [9, 8, 11, 8, 8, 9, 10, 7, 10, 10],
						  [10, 11, 10, 13, 13, 9, 11, 12, 12, 11],
						  [13, 15, 13, 16, 12, 13, 13, 11, 14, 11],
						  [12, 16, 16, 14, 12, 14, 15, 14, 16, 16]]
	data["hull"] = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 3, 1, 1, 1, 1, 3, 1, 1, 1],
					[4, 1, 4, 3, 4, 1, 5, 5, 2, 1],
					[7, 3, 8, 5, 6, 5, 6, 5, 4, 6],
					[7, 8, 3, 8, 8, 9, 9, 8, 7, 7],
					[7, 9, 9, 10, 10, 9, 11, 9, 9, 9],
					[10, 11, 11, 9, 10, 11, 12, 12, 9, 12],
					[11, 10, 11, 13, 13, 12, 13, 9, 11, 9],
					[14, 15, 13, 13, 11, 13, 13, 14, 14, 15]]
	data["triangle"] = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	                    [2, 1, 2, 1, 1, 3, 1, 1, 1, 2],
	                    [1, 2, 1, 4, 1, 1, 3, 1, 2, 2],
	                    [7, 6, 2, 1, 8, 4, 5, 5, 6, 1],
	                    [9, 9, 8, 8, 8, 9, 7, 6, 9, 6],
	                    [9, 10, 8, 8, 8, 8, 6, 10, 10, 7],
	                    [11, 11, 11, 8, 11, 10, 10, 11, 11, 11],
	                    [9, 10, 12, 5, 13, 11, 12, 10, 11, 12],
	                    [11, 13, 13, 12, 11, 14, 11, 12, 13, 12]]
	labels = {"triple-disk" : "bo", "triangle" : "g^", "hull" : "ks","duo-disk" : "r."}
	plot_res(data,labels,7)
	data = {}
	data["duo-disk"] = [[4, 1, 1, 1, 4, 4, 1, 4, 4, 1],
	                    [4, 4, 4, 1, 6, 4, 5, 1, 5, 4],
	                    [8, 1, 6, 4, 4, 6, 4, 6, 6, 4],
	                    [6, 6, 5, 5, 8, 4, 6, 4, 6, 4],
	                    [7, 9, 9, 7, 6, 7, 6, 6, 7, 8],
	                    [6, 5, 8, 8, 6, 9, 6, 6, 8, 5],
	                    [6, 8, 9, 10, 8, 9, 6, 7, 6, 8],
	                    [8, 7, 11, 9, 8, 8, 12, 8, 7, 7],
	                   [9, 11, 9, 10, 9, 8, 10, 12, 11, 9],
	                   [10, 11, 9, 10, 15, 11, 13, 11, 8, 9],
	                   [10, 11, 13, 14, 12, 11, 12, 12, 13, 10],
	                   [14, 11, 12, 13, 11, 9, 14, 14, 14, 11],
	                   [17, 14, 12, 9, 13, 10, 12, 14, 11, 14],
	                   [15, 15, 14, 14, 14, 17, 14, 13, 12, 13]]
	data["triple-disk"] = [[4, 4, 4, 4, 4, 4, 4, 4, 4, 1],
	                       [6, 4, 5, 4, 5, 5, 5, 6, 6, 5],
	                       [7, 6, 5, 5, 5, 4, 7, 7, 6, 6],
	                       [4, 7, 5, 7, 8, 9, 8, 8, 6, 5],
	                       [8, 8, 6, 8, 8, 8, 6, 5, 8, 8],
	                       [7, 9, 12, 8, 7, 11, 11, 8, 6, 7],
	                       [10, 10, 11, 13, 11, 12, 10, 8, 11, 10],
	                       [12, 11, 12, 12, 12, 12, 10, 11, 12, 11],
	                       [12, 12, 13, 13, 11, 11, 12, 11, 14, 11],
	                       [13, 14, 11, 13, 10, 14, 13, 13, 10, 14],
	                       [13, 12, 13, 14, 16, 14, 13, 14, 15, 14],
	                       [15, 15, 16, 16, 17, 19, 15, 14, 14, 15],
	                       [17, 16, 16, 16, 16, 18, 17, 18, 17, 17],
	                       [17, 19, 19, 18, 18, 17, 18, 17, 17, 18]]
	data["hull"] = [[1, 4, 1, 4, 1, 1, 1, 1, 1, 1],
                    [5, 4, 1, 1, 4, 4, 4, 6, 5, 6],
                    [7, 5, 8, 5, 6, 7, 5, 6, 7, 5],
                    [6, 5, 7, 6, 6, 10, 6, 8, 5, 8],
                    [9, 8, 7, 6, 7, 11, 7, 8, 8, 9],
                    [9, 11, 11, 9, 9, 11, 8, 9, 10, 8],
                    [8, 13, 12, 11, 11, 11, 11, 11, 12, 11],
                    [10, 8, 10, 13, 10, 13, 12, 11, 11, 12],
                    [10, 12, 12, 13, 14, 13, 12, 12, 11, 14],
                    [15, 10, 14, 13, 14, 13, 14, 11, 15, 17],
                    [14, 16, 14, 11, 15, 14, 18, 15, 14, 13],
                    [16, 14, 16, 16, 16, 15, 14, 16, 16, 14],
                    [15, 15, 18, 16, 14, 19, 17, 17, 19, 16],
                    [17, 20, 18, 19, 17, 17, 19, 16, 16, 18]]
	data["triangle"] = [[1, 1, 1, 4, 1, 4, 1, 4, 4, 4],
                        [1, 4, 6, 1, 4, 4, 7, 6, 7, 4],
                        [6, 7, 8, 9, 6, 7, 6, 4, 6, 7],
                        [7, 8, 7, 8, 8, 8, 8, 9, 6, 7],
                        [8, 7, 8, 7, 6, 10, 7, 7, 9, 9],
                        [8, 9, 10, 11, 10, 10, 9, 9, 11, 9],
                        [9, 13, 9, 11, 11, 10, 13, 11, 13, 9],
                        [10, 10, 10, 11, 10, 11, 11, 9, 12, 10],
                        [12, 12, 13, 14, 11, 13, 11, 12, 13, 11],
                        [14, 13, 12, 15, 13, 14, 13, 10, 14, 13],
                        [14, 14, 13, 14, 14, 14, 14, 14, 14, 13],
                        [14, 15, 17, 15, 14, 14, 17, 15, 13, 14],
                        [17, 17, 15, 16, 16, 16, 15, 15, 18, 16],
                        [16, 19, 14, 17, 17, 16, 16, 19, 17, 16]]
	plot_res(data,labels)
	
	"""
	data = {}
	data["triple_disk"] = [[4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                           [5, 4, 5, 4, 5, 4, 4, 4, 4, 4],
                           [5, 5, 4, 5, 4, 4, 5, 4, 4, 5],
                           [5, 6, 5, 4, 5, 6, 5, 5, 4, 6],
                           [5, 5, 5, 4, 4, 5, 6, 5, 6, 6],
                           [6, 6, 6, 5, 6, 5, 6, 6, 5, 6],
                           [6, 5, 6, 6, 6, 5, 6, 5, 4, 6],
                           [6, 6, 5, 5, 6, 6, 6, 6, 6, 6],
                           [6, 6, 6, 6, 4, 6, 6, 6, 6, 6],
                           [6, 6, 7, 6, 6, 6, 6, 7, 6, 6]]
	data["duo_disk"] = [[4, 4, 4, 4, 4, 4, 4, 4, 4, 1],
	                       [6, 4, 5, 4, 5, 5, 5, 6, 6, 5],
	                       [7, 6, 5, 5, 5, 4, 7, 7, 6, 6],
	                       [4, 7, 5, 7, 8, 9, 8, 8, 6, 5],
	                       [8, 8, 6, 8, 8, 8, 6, 5, 8, 8],
	                       [7, 9, 12, 8, 7, 11, 11, 8, 6, 7],
	                       [10, 10, 11, 13, 11, 12, 10, 8, 11, 10],
	                       [12, 11, 12, 12, 12, 12, 10, 11, 12, 11],
	                       [12, 12, 13, 13, 11, 11, 12, 11, 14, 11],
	                       [13, 14, 11, 13, 10, 14, 13, 13, 10, 14],
	                       [13, 12, 13, 14, 16, 14, 13, 14, 15, 14],
	                       [15, 15, 16, 16, 17, 19, 15, 14, 14, 15],
	                       [17, 16, 16, 16, 16, 18, 17, 18, 17, 17],
	                       [17, 19, 19, 18, 18, 17, 18, 17, 17, 18]]
	plot_res(data,labels)
	"""