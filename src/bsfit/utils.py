__version__ = "0.0.1"
__author__ = "Alberto Mittone"
__lastrev__ = "Last revision: 05/07/23"

import os
import numpy as np

		
def make_dir(fname):
	if not os.path.exists(fname):
		os.system('mkdir ' + fname)
	else:
		pass


def distribute_jobs(func,proj):
	"""
	Distribute a func over proj on different cores
	"""
	args = []
	pool_size = int(mp.cpu_count()/2)
	chunk_size = int((len(proj) - 1) / pool_size + 1)
	pool_size = int(len(proj) / chunk_size + 1)
	for m in range(pool_size):
		ind_start = int(m * chunk_size)
		ind_end = (m + 1) * chunk_size
		if ind_start >= int(len(proj)):
			break
		if ind_end > len(proj):
			ind_end = int(len(proj))
		args += [range(ind_start, ind_end)]
	#mp.set_start_method('fork')
	with closing(mp.Pool(processes=pool_size)) as p:
		out = p.map_async(func, proj)
	out.get()
	p.close()		
	p.join()
	
	
def get_angle(data):
	import scipy
	rescale = 0.01
	data2 = scipy.ndimage.zoom(data, [rescale,rescale])#, output=data, order=3, mode='constant', cval=0.0, prefilter=True, grid_mode=False)
	sy, sx = data2.shape
	# Compute the gradient matrix by taking the partial derivatives with respect to x and y
	fx = np.gradient(data2, axis=1)
	fy = np.gradient(data2, axis=0)
	div = abs(fx + fy)
	max_val = np.max(np.max(div ))

	#print(max_val)
	direction = np.argwhere(div == max_val)
	x = int(sx/2) - direction[0][1]
	y = int(sy/2) - direction[0][0]

	if(x<0 and y<0):
		print(" belongs to 3rd Quadrant.")
		shift = 180.0
	elif(x<0 and y>0):
		print(" belongs to 2st Quadrant.")
		shift = 90.0
	elif(x>0 and y>0):
		print(" belongs to 1nd Quadrant.")
		shift = 0.0
	else:
		print(" belongs to 4th Quadrant.")
		shift = 270.0

	angle_degrees = np.rad2deg(np.arctan2(direction[0][0], direction[0][1]))
	return angle_degrees + shift
