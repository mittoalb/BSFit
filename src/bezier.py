# -*- coding: utf-8 -*-
"""
Created on Thu May  4 10:38:38 2023

@author: amittone
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb
import fabio
from scipy.ndimage import rotate

#from scipy.ndimage import gaussian_filter




###############################################################################
bez_range = 10 #average on +- bez_range
safe_margin = 6 #use small margin in the crop to avoid the border slope
degree_y = 4
degree_x = 4
#angle_slope = -20
###############################################################################


def get_bezier_parameters(X, Y, degree=3):
    """ Least square qbezier fit using penrose pseudoinverse.

    Parameters:

    X: array of x data.
    Y: array of y data. Y[0] is the y point for X[0].
    degree: degree of the Bézier curve. 2 for quadratic, 3 for cubic.

    Based on https://stackoverflow.com/questions/12643079/b%C3%A9zier-curve-fitting-with-scipy
    and probably on the 1998 thesis by Tim Andrew Pastva, "Bézier Curve Fitting".
    """
    if degree < 1:
        raise ValueError('degree must be 1 or greater.')

    if len(X) != len(Y):
        raise ValueError('X and Y must be of the same length.')

    if len(X) < degree + 1:
        raise ValueError(f'There must be at least {degree + 1} points to '
                         f'determine the parameters of a degree {degree} curve. '
                         f'Got only {len(X)} points.')

    def bpoly(n, t, k):
        """ Bernstein polynomial when a = 0 and b = 1. """
        return t ** k * (1 - t) ** (n - k) * comb(n, k)
        #return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

    def bmatrix(T):
        """ Bernstein matrix for Bézier curves. """
        return np.matrix([[bpoly(degree, t, k) for k in range(degree + 1)] for t in T])

    def least_square_fit(points, M):
        M_ = np.linalg.pinv(M)
        return M_ * points

    T = np.linspace(0, 1, len(X))
    M = bmatrix(T)
    points = np.array(list(zip(X, Y)))
    
    final = least_square_fit(points, M).tolist()
    final[0] = [X[0], Y[0]]
    final[len(final)-1] = [X[len(X)-1], Y[len(Y)-1]]
    return final



def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes=50):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals



def get_angle_svd(data):  
    import scipy
    rescale = 0.1
    data2 = scipy.ndimage.zoom(data, [rescale,rescale])#, output=data, order=3, mode='constant', cval=0.0, prefilter=True, grid_mode=False)

    # Compute the gradient matrix by taking the partial derivatives with respect to x and y
    fx = np.gradient(data2, axis=1)
    fy = np.gradient(data2, axis=0)
    np.savetxt(r'C:\Users\amittone\Desktop\Work\APS\TESTS_DATA\LAURA_DATA\div.txt', fx+fy, fmt='%.18e', delimiter=' ')
    import sys
    sys.exit(0)

    #G = np.stack([fx, fy], axis=-1)

    # Compute the SVD of the gradient matrix
    U, s, Vt = np.linalg.svd(fx+fy, full_matrices=True)
    print(U.shape, s.shape,Vt.shape)
    #V = Vt.T
    # Find the direction of maximum change (i.e., the unit vector in V corresponding to the largest singular value)
    #max_index = np.argmax(s)
   # print(max(s))
    max_index = np.argmax(s)
    print(s)
    print("here",max_index)
    #direction = V[:,:,max_index]
    direction = Vt[max_index]
    angle_degrees = np.rad2deg(np.arctan2(direction[1], direction[0]))
    
    #angle_degrees = np.degrees(np.arctan2(direction[1], direction[0]))

    return angle_degrees

def get_angle_PCA(data):
    import scipy
    rescale = 0.1
    data2 = scipy.ndimage.zoom(data, [rescale,rescale])#
    
    fx = np.gradient(data2, axis=1)
    fy = np.gradient(data2, axis=0)
    
    #G = np.stack([fx, fy], axis=-1)
    
    
    cov = np.cov(fx*fy)

    # compute the eigenvectors and eigenvalues of the covariance matrix
    evals, evecs = np.linalg.eig(cov)
    print(evals.shape, evecs.shape)
    # find the index of the eigenvector corresponding to the largest eigenvalue
    max_index = np.argmax(evals)
    print(max_index)
    # get the eigenvector corresponding to the largest eigenvalue
    direction = evecs[:, max_index]
    print(direction)
    #print(direction)
    # compute the angle of the eigenvector in degrees
    angle_degrees = np.rad2deg(np.arctan2(direction[1], direction[0]))
    return angle_degrees

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
    print(direction)
    
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


points = []


#data = fabio.open(r'C:\Users\amittone\Desktop\Work\APS\TESTS_DATA\MOUSE_EYES\B5_13_07um_gap86_blow__001__pag_rec__0481.edf')
#data = fabio.open(r'C:\Users\amittone\Desktop\Work\APS\TESTS_DATA\LAURA_DATA\E41_0p72um_40keV_360_ROI1__002__pag_rec__1331.edf')
data = fabio.open(r'C:\Users\amittone\Desktop\Work\APS\TESTS_DATA\LAURA_DATA\E41_0p72um_40keV_360_ROI2__001__pag_rec__1376.edf')


#angle_slope = 
angle_slope = get_angle(data.data)
print("Angle in degrees: ", angle_slope)

data.data = rotate(data.data,angle=angle_slope)#, mode='constant',cval=m)
data.data[data.data<1e-3] = 0.0


sy,sx = data.data.shape
print(sy,sx)
###############################################################################
#AUTOCROP


from scipy import ndimage
data_sobel = abs(ndimage.sobel(data.data))
print(np.max(data_sobel))
print(np.min(data_sobel))
data_sobel[data_sobel < np.max(data_sobel)*0.1 ] = 0.0
data_sobel[data_sobel != 0.0 ] = 1.0

###############################################################################

#Get first and last non zero values
first_y = next((i for i, x in enumerate(data.data[:,int(sx/2)]) if x), None) + safe_margin
last_y = (sy-next((i for i, x in enumerate(data.data[:,int(sx/2)][::-1]) if x), None)) - safe_margin

first_x = next((i for i, x in enumerate(data.data[int(sy/2),:]) if x), None) + safe_margin
last_x = (sy-next((i for i, x in enumerate(data.data[int(sy/2),:][::-1]) if x), None)) - safe_margin 
#CROP THE DATA

first = max(first_y,first_x)
last = min(last_y,last_x)

data.data=data.data[first:last,first:last]

#fill the zeros
m = np.mean(data.data[data.data>1e-2])
data.data[data.data < 0.1*m] = 0.0


imgplot = plt.imshow(np.asarray(data.data), \
                     cmap='gray', vmin=0.07, vmax=0.2)
plt.show()
###############################################################################
###############################################################################

tmp = data.data.copy()

line = np.mean(tmp[:,int(len(tmp[:,0])/2)-bez_range:int(len(tmp[:,0])/2)+bez_range],axis=1)

ypoints = line.tolist()
xpoints = np.arange(len(ypoints)).tolist()

for i in range(len(xpoints)):
    points.append([xpoints[i],ypoints[i]])

###############################################################################
#VERTICAL CORRECTION
# Plot the original points
plt.plot(xpoints, ypoints, "ro",label='Original Points')
# Get the Bezier parameters based on a degree.
data_out_y = get_bezier_parameters(xpoints, ypoints, degree_y)
x_val_y = [x[0] for x in data_out_y]
y_val_y = [x[1] for x in data_out_y]
print(data_out_y)
# Plot the control points
plt.plot(x_val_y,y_val_y,'k--o', label='Control Points')
# Plot the resulting Bezier curve
xvals_y, yvals_y = bezier_curve(data_out_y, nTimes=len(tmp[0,:]))
#print(len(xvals))
plt.plot(xvals_y, yvals_y, 'b-', label='B Curve')
plt.legend()
plt.show()

###############################################################################
#copy matrix
tmp = data.data.copy()
#Get line for fitting
ypoints = tmp[int(len(tmp[0,:])/2),:].tolist()
line = np.mean(tmp[int(len(tmp[0,:])/2)-bez_range:int(len(tmp[0,:])/2)+bez_range,:],axis=0)
#Create list of points 
ypoints = line.tolist()
xpoints = np.arange(len(ypoints)).tolist()

#Prepare input
for i in range(len(xpoints)):
    points.append([xpoints[i],ypoints[i]])

###############################################################################
#HORIZONTAL CORRECTION
# Plot the original points
plt.plot(xpoints, ypoints, "ro",label='Original Points')
# Get the Bezier parameters based on a degree.
data_out_x = get_bezier_parameters(xpoints, ypoints, degree_x)
x_val_x = [x[0] for x in data_out_x]
y_val_x = [x[1] for x in data_out_x]
print(data_out_x)
# Plot the control points
plt.plot(x_val_x,y_val_x,'k--o', label='Control Points')
# Plot the resulting Bezier curve
xvals_x, yvals_x = bezier_curve(data_out_x, nTimes=len(tmp[:,0]))
#print(len(xvals))
plt.plot(xvals_x, yvals_x, 'b-', label='B Curve')
plt.legend()
plt.show()


#CREATE MASK 
back = np.rot90(np.tensordot(yvals_y,yvals_x,axes=0),k=2)


#Assign values and change metadata
tmp = tmp/back
data.data = np.asanyarray(tmp,dtype=np.float32)
sy, sx = data.data.shape
data.header['Dim_1'] = sy
data.header['Dim_2'] = sx

print(data.header)
#Show results
imgplot = plt.imshow(np.asarray(tmp), \
                     cmap='gray')#, vmin=np.min(data_sobel)*1, vmax=np.max(data_sobel)*0.1)
plt.show()

#ROTATE BACK THE IMAGE
data.data = rotate(data.data,angle=-angle_slope, mode='constant',cval=0.0)
data.data[data.data<1e-3] = 0.0
syr, sxr = data.data.shape
data.data = data.data[int(syr/2-sy/2):int(syr/2+sy/2),int(sxr/2-sx/2):int(sxr/2+sx/2)]


#Save EDF
data.write(r'C:\Users\amittone\Desktop\Work\APS\TESTS_DATA\LAURA_DATA\E41_0p72um_40keV_360_ROI2__001__pag_rec__1376_Corr.edf')
#data.write(r'C:\Users\amittone\Desktop\Work\APS\TESTS_DATA\LAURA_DATA\E41_0p72um_40keV_360_ROI1__002__pag_rec__1331_Corr.edf')

