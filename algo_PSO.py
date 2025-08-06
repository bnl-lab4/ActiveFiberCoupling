from motion import move
from photodiode_in import getPower
from photodiode_in import get_exposure
import time
import numpy as np
import random

#Particle swarm optimization
def run(ser,file,ch,time):

	#initialize variables
	numparticles = 15
	numiterations = 20
	inertia = [5,5]

	#arbitrary values,optimization required (but will be very difficult)
	inertiaweight = 0.25
	socialweight = 0.45
	memoryweight = 0.3

	#inertiaweight = 1
	#socialweight = 1
	#memoryweight = 1

	#swarm and particles - position, fitness
	swarmoptimal = []
	particlesoptimal = []
	lastparticlepositions = []

	#initialize particles
	for i in range(numparticles):
		#initialize position
		xposition = random.random() * 75
		zposition = random.random() * 75
		#xposition = random.randint(0,14)
		#zposition = random.randint(0,14)

		move('x',xposition,ser,ch,file)
		move('z',zposition,ser,ch,file)
		lastparticlepositions.append([xposition,zposition])

		#calculate fitness
		fitness = get_exposure(time)
		#fitness = intensity_15x15[xposition][zposition]
		
		#update memory for particle
		particlesoptimal.append([xposition,zposition,fitness])

		#move to new position - for comparison reasons
		xposition = min(14,xposition + inertia[0])
		zposition = min(14,zposition + inertia[1])
		
		move('x',xposition,ser,ch,file)
		move('z',zposition,ser,ch,file)	
		lastparticlepositions[i] = [xposition,zposition]

		#calculate fitness
		fitness = get_exposure(time)
		#fitness = intensity_15x15[xposition][zposition]
		
		#update memory
		if fitness > particlesoptimal[i][2]:
			particlesoptimal[i] = [xposition,zposition,fitness]

		#update memory for swarm - the (i+1)th particle is the ith position in this array
		maxfitness = particlesoptimal[0][2]
		optimalxpos = None
		optimalzpos = None
		for r in range(numparticles-1):
			if r > 0 and particlesoptimal[r+1][2] > maxfitness:
				optimalxpos = particlesoptimal[r+1][0]
				optimalzpos = particlesoptimal[r+1][1]
				maxfitness = particlesoptimal[r+1][2]
		swarmoptimal = [optimalxpos,optimalzpos,maxfitness]

	#Iterations
	for j in range(numiterations):
		for i in range(numparticles):
			#calculate velocity vector
			pos = [lastparticlepositions[i][0],lastparticlepositions[i][1]]
			memory = [pos[0] - (particlesoptimal[i][0]),pos[1] - (particlesoptimal[i][1])]
			social = [pos[0] - swarmoptimal[0], pos[1] - swarmoptimal[1]]
			
			velocity = [(inertiaweight * inertia[0]) + (socialweight * social[0]) + (memoryweight * memory[0]),(inertiaweight * inertia[1]) + (socialweight * social[1]) + (memoryweight * memory[1])]
			
			#give velocity to particle
			move('x',pos[0] + velocity[0],ser,ch,file)
			move('z',pos[1] + velocity[1],ser,ch,file)
		
			#calculate fitness
			fitness = get_exposure(time)
			#fitness = intensity_15x15[min(14,pos[0] + velocity[0])][min(14,pos[1] + velocity[1])]
			
			#update particle memory
			lastparticlepositions[i] = [pos[0] + velocity[0],pos[1] + velocity[1]]
			if fitness > particlesoptimal[i][2]:
				particlesoptimal[i] = [pos[0] + velocity[0],pos[1] + velocity[1],fitness]
			
		#update swarm memory
		#update memory for particle - the (i+1)th particle is the ith position in this array
		maxfitness = particlesoptimal[0][2]
		optimalxpos = None
		optimalzpos = None
		for r in range(numparticles-1):
			if r > 0 and particlesoptimal[r+1][2] > maxfitness:
				optimalxpos = particlesoptimal[r+1][0]
				optimalzpos = particlesoptimal[r+1][1]
				maxfitness = particlesoptimal[r+1][2]
		swarmoptimal = [optimalxpos,optimalzpos,maxfitness]

	print(f"x,y,fitness {swarmoptimal}")		
	return swarmoptimal

    '''
    #testing data
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
	    [0.6853, 0.6676, 0.6745, 0.6637, 0.6657, 0.6716, 0.6755, 0.6755, 0.6765, 0.6745, 0.6804, 0.66,   0.675,  0.673,  0.672]
	]
	'''
