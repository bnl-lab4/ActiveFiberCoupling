import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

'''
# Define the 2D Gaussian based on laser beam profile convention
def gaussfunc(coords, A, x0, z0, wx, wz, B):
    x, z = coords
    return A * np.exp(-2 * (((x - x0) ** 2) / wx**2 + ((z - z0) ** 2) / wz**2)) + B
'''

#Define the 2D Gaussian Profile with only a single waist radius, not two
def gaussfunc(coords,A,x0,z0,w):
	x,z = coords
	#return A * np.exp(-2 * ((x**2 + z**2)/w**2))
	return A * np.exp(-2 * (((x - x0)**2 + (z - z0)**2) / w**2))
	
def gaussfit(xpos,zpos,intensity):
	scale_factor = 20 / 75
	xpos = [x * scale_factor for x in xpos]
	zpos = [z * scale_factor for z in zpos]
	
	# Stack x and z as input for curve_fit, making it a single coordinate system
	coords = np.vstack((xpos, zpos))

	# Initial parameter guess: [A, x0, z0, wx, wz, B]
	initial_guess = [
		np.max(intensity),        # A
		xpos[intensity.index(np.max(intensity))],            # x0
		zpos[intensity.index(np.max(intensity))],            # z0
		(np.max(xpos) - np.min(xpos)) / 2,  # w
		#(np.max(zpos) - np.min(zpos)) / 2,  # wz
		#np.min(intensity)         # B
	]	
	
	# Fit the model, popt is all the optimized parameters
	popt, pcov = curve_fit(gaussfunc, coords, intensity, p0=initial_guess)
	#A_fit, x0_fit, z0_fit, wx_fit, wz_fit, B_fit = popt
	A_fit, x0_fit, z0_fit, w_fit = popt

	# Extract fitted parameters
	print("Fitted parameters:")
	print(f"A  = {A_fit:.3f}")
	print(f"x0 = {x0_fit:.3f}")
	print(f"z0 = {z0_fit:.3f}")
	print(f"w = {w_fit:.3f}")
#	print(f"wz = {wz_fit:.3f}")
#	print(f"B  = {B_fit:.3f}")

	return popt

def waistfunc(y, w0, y0, zR):
    return w0 * np.sqrt(1 + ((y - y0) / zR)**2)

def waistfit(wavelength,waists):
	#y-posi: 0,37.5,75
	#wavelength: 635nm
	
	wavelength = wavelength * (1e-3)
	ypos = [0,37.5,75]
	#ypos = [0,18.75,37.5,56.25,75]
	scale_factor = 20 / 75

	ypos = [y * scale_factor for y in ypos]
	#print(ypos)
	
	# Initial guesses: [w0, y0, zR]
	w0_guess = np.min(waists)
	y0_guess = ypos[np.argmin(waists)]
	zR_guess = abs(math.pi*(w0_guess**2)/wavelength)  # ballpark

	ypos = np.array(ypos)
	waists = np.array(waists)
	waists = waists.flatten()

	print("ypos:", ypos)
	print("waists:", waists)
	print("Initial guess:", [w0_guess, y0_guess, zR_guess])
	print("waistfunc test:", waistfunc(np.array(ypos), *[w0_guess, y0_guess, zR_guess]))

	popt, pcov = curve_fit(waistfunc,ypos,waists, p0=[w0_guess, y0_guess, zR_guess])
	w0_fit, y0_fit, zR_fit = popt
	#print(f"popt: {popt}")

	print(f"Fitted beam waist w0: {w0_fit:.3f}")
	print(f"Focal position y0: {y0_fit:.3f}")
	print(f"Rayleigh range zR: {zR_fit:.3f}")
	return popt

#Define the 1D Gaussian function
def onegaussfunc(x,A,x0,w):
    return A * np.exp(-2 * ((x-x0)**2 / w**2))

def onegaussfit(xpos,intensity):
    A_guess = np.max(intensity)
    x0_guess = intensity.index(np.max(intensity))
    w_guess = (np.max(xpos) - np.min(xpos)) /2
    popt, pcov = curve_fit(onegaussfunc,xpos,intensity,p0=[A_guess,x0_guess,w_guess])

    return popt

#this is very rudimentary, but goes through the steps at which to run things. 
def run(wavelength,xpos,zpos,intensity):
	#for i in range(3):	
	#fill in with the arrays from before that collect the respective things
	optimizedparams = gaussfit(xpos,zpos,intensity)
	#the fourth parameter in the array should be the waist radii, put these in an array
	print(f"returned waist {optimizedparams[3]}")
	
	#return optimizedparams[3]
	return optimizedparams
	#waists.append(optimizedparams[3])

	'''
	#remainder of the loop generates the graph for each run
	X, Z = np.meshgrid(xpos, zpos)

	#Generate fitted intensity
	intensity_fit = gaussfunc((X, Z), *gaussfit(xpos,zpos,intensity)).reshape(X.shape)
	
	#Plot original and fitted data
	fig, axs = plt.subplots(1, 2, figsize=(12, 5))
	
	#print("xpos:", xpos.shape)
	#print("zpos:", zpos.shape)
	#print("X:", X.shape)
	intensity = np.array(intensity)
	#print("intensity:", intensity.shape)

	#intensity_2d = intensity.reshape(Z.shape)
	side = int(math.sqrt(datapoints))
	intensity_2d = intensity.reshape((side,side))
	#im1 = axs[0].imshow(intensity, extent=[xpos.min(), xpos.max(), zpos.min(), zpos.max()], origin='lower')
	im1 = axs[0].imshow(intensity_2d, extent=[min(xpos), max(xpos), min(zpos), max(zpos)], origin='lower')
	axs[0].set_title("Original Intensity")
	plt.colorbar(im1, ax=axs[0])

	im2 = axs[1].imshow(intensity_fit, extent=[min(xpos),max(xpos), min(zpos), max(zpos)], origin='lower')
	axs[1].set_title("Fitted Gaussian")
	plt.colorbar(im2, ax=axs[1])

	plt.tight_layout()
	plt.show()
	'''
	#return waists

'''
#testing
xpos = [
	 0, 15, 30, 45, 60, 75,
	 0, 15, 30, 45, 60, 75,
	37, 37, 37, 37, 37, 37,
	25, 50, 25, 50,
	37, 10, 65
]

zpos = [
	 0, 15, 30, 45, 60, 75,
	37, 37, 37, 37, 37, 37,
	 0, 15, 30, 45, 60, 75,
	25, 25, 50, 50,
	37, 65, 10
]

intensity = [
	 0.00, 2.57, 13.13, 13.13, 2.57, 0.00,
	 1.12, 9.52, 24.37, 24.37, 9.52, 1.12,
	 1.12, 9.52, 24.37, 24.37, 9.52, 1.12,
	 7.59, 7.59, 7.59, 7.59,
	30.00, 0.18, 0.18
]

xpos = [
	0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,
	0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,
	0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,
	0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,
	0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,
	0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,
	0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,
	0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,
	0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,
	0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,
	0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,
	0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,
	0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,
	0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,
	0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,
]

zpos = []
for i in range(0, 71, 5):
    zpos.extend([i] * 15)

intensity_ = [
	0.6588,0.6539,.6598,0.648,.6461,.6422,.6441,.6569,.6304,.6324,.6373,.6157,.6248,.6206,.6333,
	.6549,.6539,.6598,.6382,.6431,.6304,.6402,.6167,.601,.6108,.6098,.6049,.6127,.6108,more
	.6108,.6382,.6373,.648,.6451,.6343,.6235,.6167,.6108,.6127,.6167,.6265,.6294,.6069,.598,more	
	.5863,.6245,.65,.6392,.6451,.6147,.6294,.6108,.6216,.6225,.6275,.6314,.6441,.6382,.610,more
	.5833,.6167,.6294,.6284,.6167,.6127,.5814,.5471,.4716,.3461,.2451,.2539,.2784,.4294,more
	.5235,.5863,.6176,.6402,.6245,.602,.5637,.4637,.2431,.0196,.5196,1.3598,1.599,1.5059,1.2078,more
	.5608,.1588,.5569,.6373,.6216,.5892,.4794,.2402,.1784,.9147,2.0059,3.6049,4.3863,4.5863,3.8
	2.6451,1.5373,.45,.651,.6176,.5608,.4843,.1637,.3098,1.0765,2.4098,3.3598,4.3676,4.8235,more
	2.6559,1.2843,.0804,.649,.6216,.5667,.499,.3735,.048,.3873,1.1314,1.8137,2.2529,2.2471,more
	.6755,.8824,.002,.5745,.649,.6382,.6078,.5627,.4853,.4137,.2353,.0461,.0637,.0588,.0216,more
	.1794,.4608,.6108,.6559,.6686,.6618,.6284,.6078,.5794,.5441,.4882,.4608,.4873,.4775,.5,more
	.6059,.6588,.6618,.6637,.6608,.6569,.6539,.65,.649,.6108,.6049,.6059,.5863,.6,.593,more
	1, .6294,.6235,.6353,.648,.6588,.6686,.6559,.6627,.6539,.65,.6529,.652,.6402,.6471,.64,more
	02, .6353,.6412,.6343,.6588,.6657,.6716,.6716,.6696,.6667,.6618,.6716,.6559,.6696,.6686,more
	0.6657,.6627,.6725,.6853,.6676,.6745,.6637,.6657,.6716,.6755,.6755,.6765,.6745,.6804,.66,more
	96, .6716,.6755,.6735,.6667
]

intensity_raw = [
  0.6588, 0.6539, 0.6598, 0.648, 0.6461, 0.6422, 0.6441, 0.6569, 0.6304, 0.6324, 0.6373, 0.6157, 0.6248, 0.6206, 0.6333,
  0.6549, 0.6539, 0.6598, 0.6382, 0.6431, 0.6304, 0.6402, 0.6167, 0.601,  0.6108, 0.6098, 0.6049, 0.6127, 0.6108,
  0.6108, 0.6382, 0.6373, 0.648,  0.6451, 0.6343, 0.6235, 0.6167, 0.6108, 0.6127, 0.6167, 0.6265, 0.6294, 0.6069, 0.598,
  0.5863, 0.6245, 0.65,   0.6392, 0.6451, 0.6147, 0.6294, 0.6108, 0.6216, 0.6225, 0.6275, 0.6314, 0.6441, 0.6382, 0.610,
  0.5833, 0.6167, 0.6294, 0.6284, 0.6167, 0.6127, 0.5814, 0.5471, 0.4716, 0.3461, 0.2451, 0.2539, 0.2784, 0.4294,
  0.5235, 0.5863, 0.6176, 0.6402, 0.6245, 0.602,  0.5637, 0.4637, 0.2431, 0.0196, 0.5196, 1.3598, 1.599,  1.5059, 1.2078,
  0.5608, 0.1588, 0.5569, 0.6373, 0.6216, 0.5892, 0.4794, 0.2402, 0.1784, 0.9147, 2.0059, 3.6049, 4.3863, 4.5863, 3.8,
  2.6451, 1.5373, 0.45,   0.651,  0.6176, 0.5608, 0.4843, 0.1637, 0.3098, 1.0765, 2.4098, 3.3598, 4.3676, 4.8235,
  2.6559, 1.2843, 0.0804, 0.649,  0.6216, 0.5667, 0.499,  0.3735, 0.048,  0.3873, 1.1314, 1.8137, 2.2529, 2.2471,
  0.6755, 0.8824, 0.002,  0.5745, 0.649,  0.6382, 0.6078, 0.5627, 0.4853, 0.4137, 0.2353, 0.0461, 0.0637, 0.0588, 0.0216,
  0.1794, 0.4608, 0.6108, 0.6559, 0.6686, 0.6618, 0.6284, 0.6078, 0.5794, 0.5441, 0.4882, 0.4608, 0.4873, 0.4775, 0.5,
  0.6059, 0.6588, 0.6618, 0.6637, 0.6608, 0.6569, 0.6539, 0.65,   0.649,  0.6108, 0.6049, 0.6059, 0.5863, 0.6,    0.593,
  1.0,    0.6294, 0.6235, 0.6353, 0.648,  0.6588, 0.6686, 0.6559, 0.6627, 0.6539, 0.65,   0.6529, 0.652,  0.6402, 0.6471,
  0.64,   0.02,   0.6353, 0.6412, 0.6343, 0.6588, 0.6657, 0.6716, 0.6716, 0.6696, 0.6667, 0.6618, 0.6716, 0.6559, 0.6696,
  0.6686, 0.6657, 0.6627, 0.6725, 0.6853, 0.6676, 0.6745, 0.6637, 0.6657, 0.6716, 0.6755, 0.6755, 0.6765, 0.6745, 0.6804,
  0.66,   0.96,   0.6716, 0.6755, 0.6735, 0.6667
]

#probably need to convert to np.array() and then flatten()
intensity_15x15 = [
    [0.6588, 0.6539, 0.6598, 0.648, 0.6461, 0.6422, 0.6441, 0.6569, 0.6304, 0.6324, 0.6373, 0.6157, 0.6248, 0.6206, 0.6333],
    [0.6549, 0.6539, 0.6598, 0.6382, 0.6431, 0.6304, 0.6402, 0.6167, 0.601,  0.6108, 0.6098, 0.6049, 0.6127, 0.6108, 0.6108],
    [0.6382, 0.6373, 0.648,  0.6451, 0.6343, 0.6235, 0.6167, 0.6108, 0.6127, 0.6167, 0.6265, 0.6294, 0.6069, 0.598,  0.5863],
    [0.6245, 0.65,   0.6392, 0.6451, 0.6147, 0.6294, 0.6108, 0.6216, 0.6225, 0.6275, 0.6314, 0.6441, 0.6382, 0.610,  0.5833],
    [0.6167, 0.6294, 0.6284, 0.6167, 0.6127, 0.5814, 0.5471, 0.4716, 0.3461, 0.2451, 0.2539, 0.2784, 0.4294, 0.5235, 0.5863],
    [0.6176, 0.6402, 0.6245, 0.602,  0.5637, 0.4637, 0.2431, 0.0196, 0.5196, 1.3598, 1.599,  1.5059, 1.2078, 0.5608, 0.1588],
    [0.5569, 0.6373, 0.6216, 0.5892, 0.4794, 0.2402, 0.1784, 0.9147, 2.0059, 3.6049, 4.3863, 4.5863, 3.8,    2.6451, 1.5373],
    [0.45,   0.651,  0.6176, 0.5608, 0.4843, 0.1637, 0.3098, 1.0765, 2.4098, 3.3598, 4.3676, 4.8235, 2.6559, 1.2843, 0.0804],
    [0.649,  0.6216, 0.5667, 0.499,  0.3735, 0.048,  0.3873, 1.1314, 1.8137, 2.2529, 2.2471, 0.6755, 0.8824, 0.002,  0.5745],
    [0.649,  0.6382, 0.6078, 0.5627, 0.4853, 0.4137, 0.2353, 0.0461, 0.0637, 0.0588, 0.0216, 0.1794, 0.4608, 0.6108, 0.6559],
    [0.6686, 0.6618, 0.6284, 0.6078, 0.5794, 0.5441, 0.4882, 0.4608, 0.4873, 0.4775, 0.5,    0.6059, 0.6588, 0.6618, 0.6637],
    [0.6608, 0.6569, 0.6539, 0.65,   0.649,  0.6108, 0.6049, 0.6059, 0.5863, 0.6,    0.593,  1.0,    0.6294, 0.6235, 0.6353],
    [0.648,  0.6588, 0.6686, 0.6559, 0.6627, 0.6539, 0.65,   0.6529, 0.652,  0.6402, 0.6471, 0.64,   0.02,   0.6353, 0.6412],
    [0.6343, 0.6588, 0.6657, 0.6716, 0.6716, 0.6696, 0.6667, 0.6618, 0.6716, 0.6559, 0.6696, 0.6686, 0.6657, 0.6627, 0.6725],
    # Final row (manually extrapolated using trend from row above)
    [0.6853, 0.6676, 0.6745, 0.6637, 0.6657, 0.6716, 0.6755, 0.6755, 0.6765, 0.6745, 0.6804, 0.66,   0.675,  0.673,  0.672]
]

waists = []
wavelength = .635
for j in range(3):
	waists.append(run(wavelength,xpos,zpos,intensity_15x15))
	intensity = [i * .50 for i in intensity]

#fill in params with respective values, the w0 is the radii array
beamparams = waistfit(wavelength,waists)

#second array item should be focal point ypos
focalpointy = beamparams[1]
'''
