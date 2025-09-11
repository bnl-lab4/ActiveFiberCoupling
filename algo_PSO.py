from motion import move
from photodiode_in import getPower
from photodiode_in import get_exposure
import time
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import statistics

from statistics import stdev
import optuna

import algo_focal_estimator
import fittingvtwo

#Particle swarm optimization
def pso(ser,file,ch,t,numparticles,numiterations,inertiaweight,socialweight,memoryweight):
	start = time.time()
	
	plt.ioff()
	plt.show()
	#initialize variables
	#numparticles = 10
	#numiterations = 15
	#inertia = [ix,iz]
	inertiax = []
	inertiaz = []
	for i in range(numparticles):
		inertiax.append(random.uniform(-10,10))
		inertiaz.append(random.uniform(-10,10))
	
	count = 1

	#arbitrary values,optimization required (but will be very difficult)
	#inertiaweight = 0.25
	#socialweight = 0.45
	#memoryweight = 0.3

	#swarm and particles - position, fitness
	swarmoptimal = []
	particlesoptimal = []
	lastparticlepositions = []
	
	xp = []
	zp = []
	
	xpos = []
	zpos = []
	powers = []

	#initialize particles
	for i in range(numparticles):
		#initialize position
		xposition = random.random() * 75
		zposition = random.random() * 75
		xpos.append(xposition)
		zpos.append(zposition)

		move('x',xposition,ser,ch,file)
		move('z',zposition,ser,ch,file)
		print(count)
		count += 1
		lastparticlepositions.append([xposition,zposition])

		#calculate fitness
		fitness = get_exposure(t)
		powers.append(fitness)
		
		#update memory for particle
		particlesoptimal.append([xposition,zposition,fitness])

		#move to new position - for comparison reasons
		xposition = min(75,xposition + inertiax[i])
		zposition = min(75,zposition + inertiaz[i])
		
		move('x',xposition,ser,ch,file)
		move('z',zposition,ser,ch,file)	
		xpos.append(xposition)
		zpos.append(zposition)
		lastparticlepositions[i] = [xposition,zposition]

		#calculate fitness
		fitness = get_exposure(t)
		powers.append(fitness)
		
		#update memory
		if fitness > particlesoptimal[i][2]:
			particlesoptimal[i] = [xposition,zposition,fitness]

		#update memory for swarm - the (i+1)th particle is the ith position in this array
		maxfitness = particlesoptimal[0][2]
		optimalxpos = particlesoptimal[0][0]
		optimalzpos = particlesoptimal[0][1]
		if i == numparticles-1:
			for r in range(numparticles):
				if r > 0 and particlesoptimal[r][2] > maxfitness:
					optimalxpos = particlesoptimal[r][0]
					optimalzpos = particlesoptimal[r][1]
					maxfitness = particlesoptimal[r][2]
			swarmoptimal = [optimalxpos,optimalzpos,maxfitness]
		
	xp.append(xpos)
	zp.append(zpos)
	xpos = []
	zpos = []
		
	#print(swarmoptimal)
	#print(particlesoptimal)
	#print(lastparticlepositions)

	#Iterations
	for j in range(numiterations):
		for i in range(numparticles):
			#calculate velocity vector
			pos = [lastparticlepositions[i][0],lastparticlepositions[i][1]]
			memory = [-pos[0] + (particlesoptimal[i][0]),-pos[1] + (particlesoptimal[i][1])]
			social = [-pos[0] + swarmoptimal[0],-pos[1] + swarmoptimal[1]]
			#r1 = random.random()
			#r2 = random.random()
			r1 = 1
			r2 = 1
			velocity = [
			(inertiaweight * inertiax[i]) + (socialweight * social[0] * r1) + (memoryweight * memory[0] * r2),
			(inertiaweight * inertiaz[i]) + (socialweight * social[1] * r1) + (memoryweight * memory[1] * r2)
			]
			
			inertiax[i] = velocity[0]
			inertiaz[i] = velocity[1]
			
			#give velocity to particle
			move('x',pos[0] + velocity[0],ser,ch,file)
			move('z',pos[1] + velocity[1],ser,ch,file)
			print(count)
			count += 1
			
			#have to clamp this for calculation purposes
			tempx = min(75,pos[0] + velocity[0])
			tempx = max(0,tempx)
			tempz = min(75,pos[1] + velocity[1])
			tempz = max(0,tempz)
			
			xpos.append(tempx)
			zpos.append(tempz)
		
			xp.append(xpos)
			zp.append(zpos)
			xpos = []
			zpos = []
			
			#calculate fitness
			fitness = get_exposure(t)
			powers.append(fitness)
			#fitness = intensity_15x15[min(14,pos[0] + velocity[0])][min(14,pos[1] + velocity[1])]
			
			#update particle memory
			lastparticlepositions[i] = [tempx,tempz]
			if fitness > particlesoptimal[i][2]:
				particlesoptimal[i] = [tempx,tempz,fitness]
			
			#update swarm memory
			#update memory for particle - the (i+1)th particle is the ith position in this array
			maxfitness = particlesoptimal[0][2]
			optimalxpos = particlesoptimal[0][0]
			optimalzpos = particlesoptimal[0][1]
			if i == numparticles-1:
				for r in range(numparticles):
					if r > 0 and particlesoptimal[r][2] > maxfitness:
						optimalxpos = particlesoptimal[r][0]
						optimalzpos = particlesoptimal[r][1]
						maxfitness = particlesoptimal[r][2]
				swarmoptimal = [optimalxpos,optimalzpos,maxfitness]
				# ======= Plotting begins here =======

		# Collect all current particle positions
		curr_x = [p[0] for p in lastparticlepositions]
		curr_z = [p[1] for p in lastparticlepositions]

		# Personal bests of all particles
		best_x = [p[0] for p in particlesoptimal]
		best_z = [p[1] for p in particlesoptimal]

		# Swarm best
		swarm_x = swarmoptimal[0]
		swarm_z = swarmoptimal[1]

		# Plotting
		plt.clf()  # Clear previous frame
		plt.scatter(curr_x, curr_z, c='black', label='Current Positions')
		plt.scatter(best_x, best_z, marker='*', c='blue', s=150, label='Particle Bests')
		plt.scatter(swarm_x, swarm_z, marker='*', c='yellow', s=200, label='Swarm Best')

		plt.xlim(0, 75)
		plt.ylim(0, 75)
		plt.title(f"Iteration {j + 1}")
		plt.xlabel("X")
		plt.ylabel("Z")
		plt.legend()
		plt.pause(10)  # Update the plot

		# ======= Plotting ends here =======
	print(f"x,z,fitness {swarmoptimal}")		
	move('x',swarmoptimal[0],ser,ch,file)
	move('z',swarmoptimal[1],ser,ch,file)
	#move('z',swarmoptimal[2],ser,ch,file)
	end = time.time()
	
	print(f"time: {end - start}")
	#return (end - start), swarmoptimal[2]
	
	plt.ioff()
	plt.show()
	return xp,zp,powers
	
def make_objective(ser, file, ch, t):
	def objective(trial):
		numparticles = trial.suggest_int('n_particles',3,15)
		numiterations = trial.suggest_int('n_iterations',5,30)
		
		if numparticles * numiterations > 200:
			raise optuna.exceptions.TrialPruned()
			
		w = trial.suggest_float('w',.3,.9)
		s = trial.suggest_float('s',.5,2.5)
		m = trial.suggest_float('m',.5,2.5)
		ix = trial.suggest_float('ix',0,5)
		iz = trial.suggest_float('iz',0,5)
		
		timeB = []
		fits = []
		
		#sample size
		n = 8
		
		for i in range(n):
			tim,fit = pso(ser,file,ch,t,numparticles,numiterations,w,s,m,ix,iz)
			timeB.append(tim)
			fits.append(fit)
		
		means = statistics.mean(timeB)
		errors = (2*statistics.stdev(timeB) / math.sqrt(len(timeB)))
		fitnessmean = statistics.mean(fits)
		fitnessstdev = statistics.stdev(fits)

		score = fitnessmean - fitnessstdev - means
		
		print(f"Trial {trial.number}, numparts {numparticles}, numits {numiterations}, w {w:.3f}, s {s:.3f}, m {m:.3f}, Accuracy {fitnessmean:.4f}, Precision {fitnessstdev:.4f}, Speed {means:.4f}, Score {score:.4f}")
		
		return score
	return objective

def runtest(ser,file,ch,t):
	#pso(ser,file,ch,t,10,15,.25,.45,.3)
	study = optuna.create_study(direction='maximize')
	study.optimize(make_objective(ser, file, ch, t),n_trials=30)
	
	print("Best Trial")
	print(f"Value: {study.best_value}")
	print(f"Params: {study.best_params}")
	
	optuna.visualization.plot_optimization_history(study).show()
	optuna.visualization.plot_param_importances(study).show()

def runfocal(ser,file,ch,t):
	print("Starting scan")
	step = 5
	ypos = [0,37.5,75]

	move('y',ypos[0],ser,0,file)
	x_voltages, z_voltages, powers = pso(ser, file,ch,t,10,15,.25,.45,.3)

	#waists will hold the fitted waists of all cross sections
	#fill in wavelength of the laser, 635 = 635nm
	waists = []
	xcenter = []
	zcenter = []
	wavelength = 635
	scale = 20/75
	
	x_voltages = [x * scale for x in x_voltages]
	z_voltages = [z * scale for z in z_voltages]

	#will try to map one cross section's data onto the gaussian function
	for j in range(3):  #the loop will try to do it three times for each of the cross section but the code would change slightly because I didn't know how that was handled before
		optimized = fittingvtwo.gaussfit(x_voltages,z_voltages,powers)
		print("fitting")
		xcenter.append(optimized[1])
		zcenter.append(optimized[2])
		waists.append(optimized[3])
		if j != 2:
		    input("Press Enter to continue")
		    move('y', ypos[j+1], ser,0,file)
		    x_voltages, z_voltages, powers = pso(ser,file,ch,t,10,15,.25,.45,.3)
		    x_voltages = [x * scale for x in x_voltages]
		    z_voltages = [z * scale for z in z_voltages]
		    
	coeffTop = algo_focal_estimator.findTopLine(zcenter,waists,ypos)
	coeffBot = algo_focal_estimator.findBottomLine(zcenter,waists,ypos)

	focalpointy = algo_focal_estimator.estimatefocal(coeffTop,coeffBot)

	print(f"change the yposition to: {focalpointy * 75/20}")

	move('y',focalpointy * 75/20,ser,0,file)
	#pso(ser,file,ch,t,15,5,.3072263918607849,2.150240064896782,1.6191719579161155,4.217906861620267,4.525493213116038)
	pso(ser,file,ch,t,10,15,.25,.45,.3)
    
def run(ser,file,ch,t):
	#xpos,zpos,powers = pso(ser,file,ch,t,14,5,.6586422167567024,2.307526156402279,1.1307698374862571)
	#pso(ser,file,ch,t,15,5,.3072263918607849,2.150240064896782,1.6191719579161155,4.217906861620267,4.525493213116038)
	pso(ser,file,ch,t,10,7,.25,.45,.3)
	#print(f"xpos: {xpos}")
	#print(f"zpos: {zpos}")
	#print(f"pows: {powers}")
	
	'''
	powers = []
	for i in range(100):
		powers.append(get_exposure(100))
		
	print(f"stdev: {statistics.stdev(powers)}")
	print(f"mean: {statistics.mean(powers)}")
	print(f"median: {statistics.median(powers)}")
	'''

def runtest(ser,file,ch,t):
	wholeB = []
	timeB = []
	fitsB = []
	fits = []
	
	#sample size
	n = 30
	
	#parts = [5,5,6,10,5,10,13,5,10]
	#its = [40,30,25,15,13,10,5,26,13]

	parts = [10,13,14,15]
	its = [10,5,5,5]
	w = [.25,.25,.6586422167567024,.3072263918607849]
	s = [.45,.45,2.307526156402279,2.150240064896782]
	m = [.3,.3,1.1307698374862571,1.6191719579161155]
	ix = [2,2,2,4.217906861620267]
	iz = [3,3,3,4.525493213116038]

	for i in range(n):
		tim,fit = algo_focal_estimator.run(ser,file)
		timeB.append(tim)
		fits.append(fit)
	wholeB.append(timeB)
	fitsB.append(fits)
	timeB = []
	fits = []
	
	for j in range(len(parts)):
		for i in range(n):
			tim,fit = pso(ser,file,ch,t,parts[j],its[j],w[j],s[j],m[j],ix[j],iz[j])
			timeB.append(tim)
			fits.append(fit)
		wholeB.append(timeB)
		fitsB.append(fits)
		timeB = []
		fits = []
	#print(f"wholeB {wholeB}")from motion import move



	
'''
#testing data
xpos = hich it doesn't (based on your data collection logic).[
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
