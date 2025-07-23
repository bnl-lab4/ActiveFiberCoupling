import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

# Define the 2D Gaussian based on laser beam profile convention
def gaussfunc(coords, A, x0, z0, wx, wz, B):
    x, z = coords
    return A * np.exp(-2 * (((x - x0) ** 2) / wx**2 + ((z - z0) ** 2) / wz**2)) + B

def gaussfit(xpos,zpos,intensity):
	# Stack x and z as input for curve_fit, making it a single coordinate system
	coords = np.vstack((xpos, zpos))

	# Initial parameter guess: [A, x0, z0, wx, wz, B]
	initial_guess = [
		np.max(intensity),        # A
		np.mean(xpos),            # x0
		np.mean(zpos),            # z0
		(np.max(xpos) - np.min(xpos)) / 2,  # wx
		(np.max(zpos) - np.min(zpos)) / 2,  # wz
		np.min(intensity)         # B
	]	
	
	# Fit the model, popt is all the optimized parameters
	popt, pcov = curve_fit(gaussfunc, coords, intensity, p0=initial_guess)
	A_fit, x0_fit, z0_fit, wx_fit, wz_fit, B_fit = popt

	# Extract fitted parameters
	print("Fitted parameters:")
	print(f"A  = {A_fit:.3f}")
	print(f"x0 = {x0_fit:.3f}")
	print(f"z0 = {z0_fit:.3f}")
	print(f"wx = {wx_fit:.3f}")
	print(f"wz = {wz_fit:.3f}")
	print(f"B  = {B_fit:.3f}")

	return popt

def waistfunc(y, w0, y0, zR):
    return w0 * np.sqrt(1 + ((y - y0) / zR)**2)

def waistfit(wavelength,waists):
	#y-posi: 0,37.5,75
	#wavelength: 635nm
	
	ypos = [0,37.5,75]
	
	# Initial guesses: [w0, y0, zR]
	w0_guess = np.min(waists)
	y0_guess = ypos[np.argmin(waists)]
	zR_guess = math.pi*(w0_guess**2)/wavelength  # ballpark

	ypos = np.array(ypos)
	waists = np.array(waists)
	waists = waists.flatten()

	print("ypos:", ypos)
	print("waists:", waists)
	print("Initial guess:", [w0_guess, y0_guess, zR_guess])
	print("waistfunc test:", waistfunc(np.array(ypos), *[w0_guess, y0_guess, zR_guess]))


	popt, pcov = curve_fit(waistfunc,ypos,waists, p0=[w0_guess, y0_guess, zR_guess])
	w0_fit, y0_fit, zR_fit = popt

	print(f"Fitted beam waist w0: {w0_fit:.3f}")
	print(f"Focal position y0: {y0_fit:.3f}")
	print(f"Rayleigh range zR: {zR_fit:.3f}")
	return popt

#this is very rudimentary, but goes through the steps at which to run things. 
def run(datapoints,wavelength,xpos,zpos,intensity):
	# Sample x and z positions (10 points each)
	#xpos = np.linspace(0, 75, 25)  # [-4.5, -3.5, ..., 4.5]
	#zpos = np.linspace(0, 75, 25)
	#wavelength = 635

	# Sample intensity array (10x10 grid), values between 0 and .2
	#intensity = np.array([
	#	0.01, 0.03, 0.06, 0.1, 0.15,
	#	0.02, 0.05, 0.1, 0.18, 0.3,
	#	0.03, 0.1, 0.2, 0.35, 0.5, 
	#	0.05, 0.15, 0.3, 0.55, 0.75, 
	#	0.08, 0.22, 0.45, 0.75, 1.0, 
	#])
	'''	
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
	'''	

	#intensity = intensity.flatten()
	#intensity = [i * 1.00 for i in intensity]


#	print(f"x_positions: ", xpos)
#	print(f"z_positions: ", zpos)
	#print("intensity shape:", intensity.shape)
	
	#optimizedparams = []
	waists = []
	
	#for i in range(3):	
	#fill in with the arrays from before that collect the respective things
	optimizedparams = gaussfit(xpos,zpos,intensity)
	#the fourth parameter in the array should be the waist radii, put these in an array
	
	waists.append(optimizedparams[3])

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
	
	return waists
	
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

waists = []
wavelength = 635
for j in range(3):
	waists.append(run(25,wavelength,xpos,zpos,intensity))
	intensity = [i * .50 for i in intensity]

#fill in params with respective values, the w0 is the radii array
beamparams = waistfit(wavelength,waists)

#second array item should be focal point ypos
focalpointy = beamparams[1]
