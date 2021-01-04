# Create data
import pandas as pd 
import numpy as np 

# Case 1 
classSize1 = 100
classSize2 = 100
# classSize3 = 100
sizeObservations = classSize1 + classSize2 #+ classSize3

location1 = -5
location2 = -5
location3 = 5
location4 = 5
# location3 = 6

### Spikes Even and Uneven
observations = [location1]*classSize1 + [location2]*classSize2 #+ [location3]*classSize3
assignments = [0]*classSize1 + [1]*classSize2 #+ [2]*classSize3

observations = np.array(observations)
assignments = np.array(assignments)

### Spikes with Gaussian Noise
# variance1 = 1.0
# variance2 = 1.0
# observations1 = np.random.normal(location1, variance1, classSize1)
# observations2 = np.random.normal(location2, variance2, classSize2)
# observations = np.concatenate((observations1, observations2))
# assignments = [0]*classSize1 + [1]*classSize2


dfAssignments = pd.Series(assignments, dtype=np.int16)
dfAssignments.to_csv("assignments.csv",header=None, index=False)

dfObservations = pd.Series(observations,dtype=np.float32)
dfObservations.to_csv("y.csv",header=None, index=False)
dfObservations.to_csv("Y.csv",header=None, index=False)

dfCenters = pd.Series([location1, location2])#, location3])
dfCenters.to_csv("x.csv", header=None, index=False)