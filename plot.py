import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from collections import Counter
import os 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str)
args = parser.parse_args()
PATH = args.path
# Read data 
centers = pd.read_csv(os.path.join(PATH,"x_true.csv"), header=None)
numberClusters = len(centers.values[0])

x = pd.read_csv(os.path.join(PATH,"x_estimate.csv"), header=None)
s_x = pd.read_csv(os.path.join(PATH,"s_x.csv"), header=None)
s_z = pd.read_csv(os.path.join(PATH,"s_z.csv"), header=None)
y = pd.read_csv(os.path.join(PATH,"y_observed.csv"), header=None)
assignments = pd.read_csv(os.path.join(PATH,"assignments.csv"), header=None)

fig, axs = plt.subplots(2,2)
axs = axs.ravel()
fig.suptitle("Experiment")

#Cluster Center Estimate
axs[0].plot(x.iloc[:,0], label="Estimate Center 1")
axs[0].plot(x.iloc[:,1], label="Estimate Center 2")
axs[0].axhline(y=float(centers.values[0,0]), color="g", label="True Center 1")
axs[0].axhline(y=float(centers.values[0,1]), color="r", label="True Center 2")
axs[0].title.set_text("Cluster Center Estimates (x_estimate)")
axs[0].set_xlabel("Iterations")
axs[0].legend()

# Observations 
y = y.values[0]
assignments = assignments.values[0]

scatterPlotY = axs[1].scatter(np.array(range(0,len(y))),y,label="x",c=assignments,cmap='jet')
axs[1].set_xlabel("y_i")
axs[1].title.set_text("Observations y")
axs[1].legend(scatterPlotY.legend_elements()[0],["0: %i"%list(assignments).count(0),"1: %i"%list(assignments).count(1)])


#Plot variances 
axs[2].plot(s_x.iloc[:,0], label="Variance s_x0")
axs[2].plot(s_x.iloc[:,1], label="Variance s_x1")
axs[2].title.set_text("Variance S_xi")
axs[2].set_xlabel("Iteration")
axs[2].legend()

# Plot assignments
s_z = np.array(s_z.iloc[-1,:]).reshape(len(y),numberClusters)

# Get assignment estimate 
assignment_estimate = np.argwhere(s_z>0.0)
s_z_assignment = assignment_estimate[:,1]
s_z_it = assignment_estimate[:,0]
# Nonzero s_z values
s_z_estimate = s_z[np.where(s_z>0.0)]

# If there is more than one non zero values for one index, mark it as unassigned
unassigned = list(set([x for x in list(s_z_it) if list(s_z_it).count(x) > 1]))
unassigned_rows = []
for r in unassigned:
    unassigned_rows = unassigned_rows + list(np.where(assignment_estimate==unassigned[0])[0])

counts_old = Counter(s_z_assignment)
for r in unassigned_rows:
    s_z_assignment[r] = numberClusters 
counts = Counter(s_z_assignment)
counts[numberClusters] = len(unassigned)



scatterPlot = axs[3].scatter(s_z_it, s_z_estimate,c=s_z_assignment,cmap='jet')
axs[3].legend(scatterPlot.legend_elements()[0],["0: %i"%counts[0],"1: %i"%counts[1],"Unassigned: %i"%counts[numberClusters]])
axs[3].title.set_text("S_z values")
axs[3].set_ylabel("Value of S_Zi")
axs[3].set_xlabel("Observation i")
plt.show()

