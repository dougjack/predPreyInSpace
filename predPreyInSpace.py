# Coupled intransitive loops
# John Vandermeer
# Doug Jackson
# Senay Yitbarek
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors
import time as tM

################################################################################################
# Constants
################################################################################################
workingDir = "/Users/djackson/Documents/JohnV/intransitiveLoops_2016/predPreyInSpace/"

enablePlots = True
endTime = 5000
plotInterval = 100
plotStartTime = 1

# Control execution of the space-setting system (system 2)
transientTime = 50
tau = 5000

# Arena size
nRow = 100
nCol = 100
numCells = nRow*nCol

# Approximate initial populations
initNumPrey = 1000*(nRow*nCol)/(100*100)
initNumPred1 = 9000*(nRow*nCol)/(100*100)
initNumPred2 = 5000*(nRow*nCol)/(100*100)
initNumPrey2 = 500*(nRow*nCol)/(100*100)
initNumPred3 = 500*(nRow*nCol)/(100*100)

# Probabilities
# First system
preyMigration = 0.99
pred1death = 0.1
pred2death = 0.1
pred1attack = 0.9
pred2attack = 0.85

# Second system
prey2migration = 0.25
pred3death = 0.8
pred3attack = 0.8

# Probability of a point being swapped during the small world random scrambling
# Predator 2 gets scrambled (i.e., it has long-distance migration)
probScramble = 0.04

# Predator bias term
# 1 = only predator 1, 0 = only predator 2
predBias = 0

# Settings for brute-force search for parameter values that allow pred1 and pred2 to coexist
# sweepType = tau, random, or fig1
sweepType = "tau"
replicates = 100

tauStart = 1
tauStop = 40
tauStep = 1
taus = np.arange(tauStart, tauStop, step=tauStep)
taus = np.repeat(taus, replicates)
numRuns = len(taus)

# Settings for the Figure 1 sweep
#preyMigrationStart = 0.01
#preyMigrationStep = 0.01
#preyMigrationEnd = 0.5
#pred1deathStart = 0.1
#pred1deathStep = 0.1
#pred1deathEnd = 1
#pred1attackStart = 0.1
#pred1attackStep = 0.1
#pred1attackEnd = 1.0

#preyMigrations = np.arange(preyMigrationStart, preyMigrationEnd + preyMigrationStep, 
#                           preyMigrationStep)
#preyMigrations = [0.08, 0.05]
#pred1deaths = np.arange(pred1deathStart, pred1deathEnd + pred1deathStep, pred1deathStep)
#pred1attacks = np.arange(pred1attackStart, pred1attackEnd + pred1attackStep, pred1attackStep)
#params = tuple(itertools.product(preyMigrations, pred1deaths, pred1attacks, np.arange(0, replicates)))
#numRuns = len(params)

#numRuns = 100
#preyMigrationMin = 0
#preyMigrationMax = 0.3
#pred1attackMin = 0.8
#pred1attackMax = 1.0
#probScrambleMin = 0
#probScrambleMax = 0.15

# Disable sweeps by setting numRuns = 1
numRuns = 1

abortOnExtinct = False

initVectorLen = int(np.min([endTime, 100000]))

# Background habitat
enableBackgroundHabitat = False
backgroundHabitatFile = os.path.join(workingDir, "backgroundHabitat", "background_8228_2_clusters_edited.csv")

preyCol = "#f1a340"
pred1col = "#5e3c99"
pred2col = "#5b905b"
prey2col = "#386cb0"
pred3col = "#519cff"
colorMap = ["white", preyCol, pred1col, pred2col, "black", prey2col, pred3col]

# For reference: command to close all plot windows is plt.close("all")

################################################################################################
# Functions
################################################################################################
# Calculate the number of neighbors in the Moore neighborhood. Assumes a torus
def calcMoore(occMat):
    
    [nRow, nCol] = occMat.shape
    
    # Create shifted versions of the occupancy matrix. Names refer to the relative
    # locations of the neighbors
    upper = np.roll(occMat, 1, axis=0)
    lower = np.roll(occMat, -1, axis=0)
    left = np.roll(occMat, 1, axis=1)
    right = np.roll(occMat, -1, axis=1)
    upperRight = np.roll(right, 1, axis=0)
    upperLeft = np.roll(left, 1, axis=0)
    lowerRight = np.roll(right, -1, axis=0)
    lowerLeft = np.roll(left, -1, axis=0)
    
    # Add up the neighbors
    neighborMat = occMat + upper + lower + left + right + upperRight + upperLeft + \
        lowerRight + lowerLeft
    
    return(neighborMat)

# Randomly swap values in the occupancy matrix with some probability
def scrambleMat(occMat, probScramble):

    # Make a local copy so the original isn't affected
    occMat = occMat.copy()
    
    [nRow, nCol] = occMat.shape
    
    # Determine which points in the occupancy matrix should be swapped
    randMat = np.random.rand(nRow, nCol)
    scramble = randMat<probScramble
    
    # Arrange the points to be swapped in random order
    scrambleRow = np.where(scramble)[0]
    scrambleCol = np.where(scramble)[1]
    scrambleIndices = np.arange(0, len(scrambleRow))
    np.random.shuffle(scrambleIndices)
    scrambleIndices = list(scrambleIndices)
    
    # Loop through the points to be swapped, swapping the values of pairs of points
    while len(scrambleIndices)>1:
        
        # Grab a pair of points from the list
        index1 = scrambleIndices.pop()
        index2 = scrambleIndices.pop()
        
        # Swap the points
        tempValue = occMat[scrambleRow[index1], scrambleCol[index1]]
        occMat[scrambleRow[index1], scrambleCol[index1]] = \
              occMat[scrambleRow[index2], scrambleCol[index2]] 
        occMat[scrambleRow[index2], scrambleCol[index2]] = tempValue
    
    return(occMat)  
    
################################################################################################
# Run
################################################################################################
os.chdir(workingDir)

# Make a color map of fixed colors
# Order of colors: (background, prey, pred1, pred2, unavailable, prey2, pred3)
#cmap = colors.ListedColormap(["white", "red", "blue", "green", "black", "magenta", "cyan"])
cmap = colors.ListedColormap(colorMap)

# Create a figure for the lattice plot
images = []

# Load background habitat file
if enableBackgroundHabitat:
    backgroundHabitat = np.array(pd.read_csv(backgroundHabitatFile, header=None))
    backgroundHabitat = backgroundHabitat==1
else:
    backgroundHabitat = np.zeros([nRow, nCol]).astype(int)
    backgroundHabitat = backgroundHabitat==1
    
endTime = int(endTime)
searchNum = str(np.random.randint(100000))
runs = pd.DataFrame()
executionStartTime = tM.time()
for runNum in range(numRuns):
    
    print("======================================================================")
    print("Starting run:", runNum)
    print("======================================================================")
    
    if numRuns>1 and sweepType=="random":
        preyMigration = np.round(np.random.uniform(preyMigrationMin, preyMigrationMax), decimals=2)
        pred1death = np.round(np.random.rand(), decimals=2)
        pred2death = np.round(np.random.rand(), decimals=2)
        pred1attack = np.round(np.random.uniform(pred1attackMin, pred1attackMax), decimals=2)
        pred2attack = np.round(np.random.rand(), decimals=2)
        probScramble = np.round(np.random.uniform(probScrambleMin, probScrambleMax), decimals=2)
    
    if numRuns>1 and sweepType=="tau":
        tau = taus[runNum]
    
    if numRuns>1 and sweepType=="fig1":
        preyMigration, pred1death, pred1attack, replicate = params[runNum]

    # Initialize the occupancy matrices
    prey = np.random.choice([0, 1], p=[1-(initNumPrey/numCells), initNumPrey/numCells], 
                            size=[nRow, nCol])
    pred1 = np.random.choice([0, 1], p=[1-(initNumPred1/numCells), initNumPred1/numCells], 
                            size=[nRow, nCol])
    pred2 = np.random.choice([0, 1], p=[1-(initNumPred2/numCells), initNumPred2/numCells], 
                            size=[nRow, nCol])
    prey2 = np.random.choice([0, 1], p=[1-(initNumPrey2/numCells), initNumPrey2/numCells], 
                      size=[nRow, nCol])
    pred3 = np.random.choice([0, 1], p=[1-(initNumPred3/numCells), initNumPred3/numCells], 
                            size=[nRow, nCol])      
    
    # Apply the background habitat (1=unattainable)
    if enableBackgroundHabitat:
        prey[backgroundHabitat] = 0
        pred1[backgroundHabitat] = 0
        pred2[backgroundHabitat] = 0

    empty = ((prey + pred1 + pred2 + prey2 + pred3)==0).astype(int)
    emptySys2 = ((prey2 + pred3)==0).astype(int)
    
    # Initialize vectors to store the populations
    numPrey = np.zeros([initVectorLen+1, 1])
    numPred1 = np.zeros([initVectorLen+1, 1])
    numPred2 = np.zeros([initVectorLen+1, 1])
    numPrey2 = np.zeros([initVectorLen+1, 1])
    numPred3 = np.zeros([initVectorLen+1, 1])
    numEmpty = np.zeros([initVectorLen+1, 1])
    grewVectors = False
    
    # Main loop
    times = np.arange(0, endTime+1)
    for time in times:
        
        if time%1000==0:
            print("runNum:", runNum, ", tau:", tau, ", time:", time, 
                  ", execution time:", tM.time() - executionStartTime)
            
        # Store the previous time step
        prevPrey = prey.copy()
        prevPred1 = pred1.copy()
        prevPred2 = pred2.copy()
        prevPrey2 = prey2.copy()
        prevPred3 = pred3.copy()
        prevEmpty = ((prevPrey + prevPred1 + prevPred2 + prevPrey2 + prevPred3)==0).astype(int)
        prevEmptySys2 = ((prevPrey2 + prevPred3)==0).astype(int)
        
        # Calculate the numbers of neighbors
        neighborsPrey = calcMoore(prevPrey)
        neighborsPred1 = calcMoore(prevPred1)
        neighborsPred2 = calcMoore(scrambleMat(prevPred2, probScramble)) # calcMoore(prevPred2)
        neighborsPrey2 = calcMoore(prevPrey2)
        neighborsPred3 = calcMoore(prevPred3)
            
        if time<transientTime or time%tau==0:
            # Second system
            # Predator 3 death
            randMat = np.random.rand(nRow, nCol)
            pred3[randMat<pred3death] = 0
                 
            # Prey2 -> predator 3 (predator 3 attack)
            randMat = np.random.rand(nRow, nCol)
            probAttack = neighborsPred3*pred3attack*prevPrey2
            attacked = randMat<probAttack
            prey2[attacked]=0
            pred3[attacked]=1
                 
            # Empty space -> prey2 (prey2 migration)
            randMat = np.random.rand(nRow, nCol)
            probMigration = neighborsPrey2*prey2migration*prevEmptySys2
            migration = randMat<probMigration
            emptySys2[migration] = 0
            prey2[migration] = 1
            
        # First system
        # Predator 2 death
        randMat = np.random.rand(nRow, nCol)
        pred2[randMat<pred2death] = 0
       
        # Predator 1 death
        randMat = np.random.rand(nRow, nCol)
        pred1[randMat<pred1death] = 0
             
        # Prey -> predator 1 (predator 1 attack)
        randMat = np.random.rand(nRow, nCol)
        probAttack = neighborsPred1*pred1attack*prevPrey
        attacked = randMat<probAttack
        prey[attacked]=0
        pred1[attacked]=1
        
        # Prey -> predator 2 (predator 2 attack)
        randMat = np.random.rand(nRow, nCol)
        randMat2 = np.random.rand(nRow, nCol)
        probAttack = neighborsPred2*pred2attack*prevPrey
        attacked = np.logical_and(randMat<probAttack, randMat2>predBias)
        prey[attacked]=0
        pred2[attacked]=1
        
        # Empty space -> prey (prey migration)
        randMat = np.random.rand(nRow, nCol)
        probMigration = neighborsPrey*preyMigration*prevEmpty
        migration = randMat<probMigration
        empty[migration] = 0
        prey[migration] = 1
        
        # Transform VP1P2 (both predators) to empty space
        bothPreds = (pred1 + pred2)>1
        pred1[bothPreds] = 0
        pred2[bothPreds] = 0
        empty[bothPreds] = 1
            
        # Cells with either prey2 or pred3 are inaccessible to the first system
        sys2occ = (prey2 + pred3)>0
        prey[sys2occ] = 0
        pred1[sys2occ] = 0
        pred2[sys2occ] = 0
        empty = ((prey + pred1 + pred2 + prey2 + pred3)==0).astype(int)
        emptySys2 = ((prey2 + pred3)==0).astype(int)
             
        # Apply the background habitat (1=unattainable)
        if enableBackgroundHabitat:
            prey[backgroundHabitat] = 0
            pred1[backgroundHabitat] = 0
            pred2[backgroundHabitat] = 0
            empty = ((prey + pred1 + pred2 + prey2 + pred3)==0).astype(int)
            emptySys2 = ((prey2 + pred3)==0).astype(int)
        
        # Grow the vectors, if necessary
        if time>initVectorLen and not grewVectors:
            numPrey = np.concatenate([numPrey, np.zeros([endTime-initVectorLen, 1])])
            numPred1 = np.concatenate([numPred1, np.zeros([endTime-initVectorLen, 1])])
            numPred2 = np.concatenate([numPred2, np.zeros([endTime-initVectorLen, 1])])
            numPrey2 = np.concatenate([numPrey2, np.zeros([endTime-initVectorLen, 1])])
            numPred3 = np.concatenate([numPred3, np.zeros([endTime-initVectorLen, 1])])
            numEmpty = np.concatenate([numEmpty, np.zeros([endTime-initVectorLen, 1])])
            grewVectors = True
            
        # Record the population sizes
        numPrey[time] = np.sum(prey)
        numPred1[time] = np.sum(pred1)
        numPred2[time] = np.sum(pred2)
        numPrey2[time] = np.sum(prey2)
        numPred3[time] = np.sum(pred3)
        numEmpty[time] = np.sum(empty)
        
        # Plot the lattice
        if time%plotInterval==0 and time>plotStartTime and enablePlots:
            images.append([plt.pcolormesh(prey + 2*pred1 + 3*pred2 + 4*backgroundHabitat + 5*prey2 + 6*pred3, 
                                          vmin=0, vmax=7, cmap=cmap)])

        # End the run if either of the predators has gone extinct
        if (numPred1[time]==0 or numPred2[time]==0) and abortOnExtinct:
            break
        
    runs = runs.append(pd.DataFrame([{"preyMigration":preyMigration,
                                  "pred1death":pred1death,
                                  "pred2death":pred2death,
                                  "pred1attack":pred1attack,
                                  "pred2attack":pred2attack,
                                  "probScramble":probScramble,
                                  "tau":tau,
                                  "numPrey":numPrey[time][0], "numPred1":numPred1[time][0], 
                                  "numPred2":numPred2[time][0], "time":time}]),
                                    ignore_index=True)        
    # Save the runs dataframe
    if (runNum%100==0 or runNum==(numRuns-1)) and numRuns>1:
        runs.to_csv("runs_" + searchNum + ".csv", index=False)

# Write the matrices to files for plotting
prey = pd.DataFrame(prey)
prey.to_csv("prey.csv", index=False)
pred1 = pd.DataFrame(pred1)
pred1.to_csv("pred1.csv", index=False)
pred2 = pd.DataFrame(pred2)
pred2.to_csv("pred2.csv", index=False)
prey2 = pd.DataFrame(prey2)
prey2.to_csv("prey2.csv", index=False)
pred3 = pd.DataFrame(pred3)
pred3.to_csv("pred3.csv", index=False)


# Write the empty matrix to a file so power laws can be calculated
emptyDF = pd.DataFrame(empty)
emptyDF.to_csv("empty.csv", index=False)

# Combine the results into a dataframe so they can be saved to a file
results = pd.DataFrame({"time":times[0:len(numPrey)], "numPrey":numPrey.flatten(),
                        "numPred1":numPred1.flatten(), "numPred2":numPred2.flatten(),
                        "numPrey2":numPrey2.flatten(), "numPred3":numPred3.flatten()},
    columns=["time", "numPrey", "numPred1", "numPred2", "numPrey2", "numPred3"])
results.to_csv("results.csv", index=False)

if enablePlots:
    # Plot the populations
    fig, ax = plt.subplots(figsize=[10, 10]) 
    ax.plot(times, numPrey, colorMap[1], linewidth=2, alpha=0.75) 
    ax.plot(times, numPred1, color=colorMap[2], linewidth=2, alpha=0.75)
    ax.plot(times, numPred2, color=colorMap[3], linewidth=2, alpha=0.75)
    ax.plot(times, numPrey2, colorMap[5], linewidth=2, alpha=0.75)
    ax.plot(times, numPred3, colorMap[6], linewidth=2, alpha=0.75)
    ax.set_xlabel("time step")
    ax.set_ylabel("population")
    ax.legend(["prey", "predator 1", "predator 2", "prey 2", "predator 3"], fontsize=8,
              bbox_to_anchor=(0.75, 1.05), ncol=5)
        
    # Plot the lattice
    fig2 = plt.figure()
    imAni = animation.ArtistAnimation(fig2, images, interval=200, repeat_delay=500,
                                      blit=True)
    plt.show()