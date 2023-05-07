__version__ = "0.0.1"
__author__ = "Alberto Mittone"
__lastrev__ = "Last revision: 05/07/23"

import argparse
import sys

from bezier import get_bezier_parameters, bezier_curve 
import glob
from utils import make_dir, get_angle
import BSlogging
from scipy import ndimage
import fabio
from scipy.ndimage import rotate
import numpy as np	
	
def main():

	parser = argparse.ArgumentParser(description='CT background removal using Bezier surface fit.')
	parser.add_argument('--bzrange', type=int, help='Bezier line range +/-',default=10)
	parser.add_argument('--smargin', type=int, help='Safe_margin - cuts the side of the CT',default=6)
	parser.add_argument('--dy', type=int, help='Bezier degree in y',default=4)
	parser.add_argument('--dx', type=int, help='Bezier degree in x',default=4)
	parser.add_argument('--fname', type=str, help='Input file name',default=None)
	parser.add_argument('--show_plots', type=bool, help='Show plots and images',default=True)
	#parser.add_argument('--sfile', type=int, help='use a single file',default=False)
	parser.add_argument('--verbose', type=int, help='Show extra info',default=0)
	args = parser.parse_args()




	###############################################################################
	bez_range = args.bzrange #average on +- bez_range
	safe_margin = args.smargin #use small margin in the crop to avoid the border slope
	degree_y = args.dy
	degree_x = args.dx
	###############################################################################
	
	if args.fname == None:
		BSlogging.logger.info('Specify the file name\n')
		print('Specify the file name\n')
		sys.exit()
	
	if args.show_plots:	
		import matplotlib.pyplot as plt
	
	#create outname
	tmp = args.fname.split('/')
	folder = '/'.join(tmp[:-1])
	tmp.insert(-1,'corrected')
 	
	ndir = '/'.join(tmp[:-1])
	make_dir(ndir)
	ext = args.fname.split('.')[-1]

	total = glob.glob(folder + "/*." + ext)
	total.sort()
	
	if int(len(total)) == 0:
		print('No files found.')
		sys.exit() 
	
	for i in total:
		#READ DATA
		print("Processing", i,'\n')
		data = fabio.open(i)

		pointsx = []	
		pointsy = []
		#angle_slope = 
		angle_slope = get_angle(data.data)
		print("Angle in degrees: ", angle_slope)

		data.data = rotate(data.data,angle=angle_slope)#, mode='constant',cval=m)
		data.data[data.data<1e-3] = 0.0


		sy,sx = data.data.shape
		#print(sy,sx)
		###############################################################################
		#AUTOCROP



		data_sobel = abs(ndimage.sobel(data.data))
		#print(np.max(data_sobel))
		#print(np.min(data_sobel))
		data_sobel[data_sobel < np.max(data_sobel)*0.1 ] = 0.0
		data_sobel[data_sobel != 0.0 ] = 1.0

		ssy, ssx = data_sobel.shape
		###############################################################################

		#Get first and last non zero values
		first_y = next((i for i, x in enumerate(data.data[:,int(sx/2)]) if x), None) + safe_margin
		last_y = (sy-next((i for i, x in enumerate(data.data[:,int(sx/2)][::-1]) if x), None)) - safe_margin

		first_x = next((i for i, x in enumerate(data.data[int(sy/2),:]) if x), None) + safe_margin
		last_x = (sx-next((i for i, x in enumerate(data.data[int(sy/2),:][::-1]) if x), None)) - safe_margin 
		#CROP THE DATA

		first = max(first_y,first_x)
		last = min(last_y,last_x)

		#CREATE OVELAPPING CROSS
		
		proc_mask = np.zeros_like(data_sobel)
		proc_mask += 0.1
		
		proc_mask[first:last,int(ssx/2)] = 1.0
		proc_mask[int(ssy/2),first:last] = 1.0
		
		
		#print(np.max(data_sobel),np.min(data_sobel))
		if args.verbose and args.show_plots:
			plt.title('Sobel range')
			plt.imshow(data.data[first-safe_margin:last+safe_margin,first-safe_margin:last+safe_margin]*      \
					proc_mask[first-safe_margin:last+safe_margin,first-safe_margin:last+safe_margin], \
					     cmap='gray', 
					     vmin=np.min(data.data[first-safe_margin:last+safe_margin,first-safe_margin:last+safe_margin]), 
					     vmax=np.max(data.data[first-safe_margin:last+safe_margin,first-safe_margin:last+safe_margin])*0.1)
			#plt.imshow(proc_mask[first-safe_margin:last+safe_margin,first-safe_margin:last+safe_margin], \
			#		     cmap='YlOrBr', vmin=1.0, vmax=1.0, alpha=0.5)		     
			plt.show()

		data.data=data.data[first:last,first:last]

		#fill the zeros
		m = np.mean(data.data[data.data>1e-2])
		data.data[data.data < 0.1*m] = 0.0

		###############################################################################
		###############################################################################
		#VERTICAL CORRECTION
		tmp = data.data.copy()

		liney = np.mean(tmp[:,int(len(tmp[:,0])/2)-bez_range:int(len(tmp[:,0])/2)+bez_range],axis=1)

		ypointsy = liney.tolist()
		xpointsy = np.arange(len(ypointsy)).tolist()

		for i in range(len(xpointsy)):
		    pointsy.append([xpointsy[i],ypointsy[i]])

		# Get the Bezier parameters based on a degree.
		data_out_y = get_bezier_parameters(xpointsy, ypointsy, degree_y)
		x_val_y = [x[0] for x in data_out_y]
		y_val_y = [x[1] for x in data_out_y]
		xvals_y, yvals_y = bezier_curve(data_out_y, nTimes=len(tmp[:,0]))

		###############################################################################
		#HORIZONTAL CORRECTION
		tmp = data.data.copy()
		
		#ypoints = tmp[int(len(tmp[0,:])/2),:].tolist()
		linex = np.mean(tmp[int(len(tmp[0,:])/2)-bez_range:int(len(tmp[0,:])/2)+bez_range,:],axis=0)	
		ypointsx = linex.tolist()
		xpointsx = np.arange(len(ypointsx)).tolist()

		#Prepare input
		for i in range(len(xpointsx)):
		    pointsx.append([xpointsx[i],ypointsx[i]])

		# Get the Bezier parameters based on a degree.
		data_out_x = get_bezier_parameters(xpointsx, ypointsx, degree_x)
		x_val_x = [x[0] for x in data_out_x]
		y_val_x = [x[1] for x in data_out_x]		
		xvals_x, yvals_x = bezier_curve(data_out_x, nTimes=len(tmp[0,:]))

		#CREATE MASK 
		back = np.rot90(np.tensordot(yvals_y,yvals_x,axes=0),k=2)


		#Assign values and change metadata
		tmp = tmp/back
		
		#SHOW RESULTS
		if args.show_plots:	
			
			plt.subplot(2,2,1)
			plt.title('Original')
			plt.imshow(np.asarray(data.data), \
					     cmap='gray')#, vmin=np.min(data_sobel)*1, vmax=np.max(data_sobel)*0.1)
			plt.subplot(2,2,2)		
			plt.title('Corrected')			     
			plt.imshow(np.asarray(tmp), \
					     cmap='gray')#, vmin=np.min(data_sobel)*1, vmax=np.max(data_sobel)*0.1)
			
			plt.subplot(2,2,3)
			plt.title('Vertical correction')	
			plt.plot(xpointsy, ypointsy, "ro",label='Original Points')
			plt.plot(x_val_y,y_val_y,'k--o', label='Control Points')
			plt.plot(xvals_y, yvals_y, 'b-', label='B Curve')
			plt.legend()

			plt.subplot(2,2,4)
			plt.title('Horizontal correction')	
			plt.plot(xpointsx, ypointsx, "ro",label='Original Points')
			plt.plot(x_val_x,y_val_x,'k--o', label='Control Points')
			plt.plot(xvals_x, yvals_x, 'b-', label='B Curve')	
			plt.legend()
			
			plt.show()#sblock=True)
		
		
		data.data = np.asanyarray(tmp,dtype=np.float32)
		sy, sx = data.data.shape
		data.header['Dim_1'] = sy
		data.header['Dim_2'] = sx

		BSlogging.logger.info(data.header)
		#Show results

		#ROTATE BACK THE IMAGE
		data.data = rotate(data.data,angle=-angle_slope, mode='constant',cval=0.0)
		data.data[data.data<1e-3] = 0.0
		syr, sxr = data.data.shape
		data.data = data.data[int(syr/2-sy/2):int(syr/2+sy/2),int(sxr/2-sx/2):int(sxr/2+sx/2)]


		#OUTNAME
		tmp = args.fname.split('.')
		tmp.insert(-1,'corrected')
		outname = '_'.join(tmp[:-1]) + '.' + tmp[-1]	
		
		tmp = outname.split('/')
		folder = '/'.join(tmp[:-1])
		tmp.insert(-1,'corrected')
		#tmp = tmp[:-1]
		outname  = '/'.join(tmp)# + outname.split('/')[-1]
		data.write(outname)

if __name__ == "__main__":
	
	main()

