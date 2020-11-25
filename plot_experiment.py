import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from collections import Counter
import os 
import argparse
import scipy.stats as stats


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
costX = pd.read_csv(os.path.join(PATH,"costX.csv"), header=None)
costZ = pd.read_csv(os.path.join(PATH,"costZ.csv"), header=None)

config = pd.read_csv(os.path.join(PATH,"config.csv"),header=None,index_col=0)
r_x = config.loc["r_x"][1]
r_z = config.loc["r_z"][1]
clusterVariance1 = config.loc["clusterVariance1"][1]
clusterVariance2 = config.loc["clusterVariance2"][1]
beta = config.loc["beta"][1]

priorCodes = {0:"SNUV", 1:"HUBER", 2:"L1", 3:"NUV"}
priorX = priorCodes[config.loc["lossFunctionX"][1]]
priorZ = priorCodes[config.loc["lossFunctionZ"][1]]


fig, axs = plt.subplots(2,3)
# axs = axs.ravel()
fig.suptitle("Experiment: r_x %.4f, r_z %.4f, var_x1 %.2f, var_x2 %.2f, beta %.4f"%(r_x,r_z, clusterVariance1, clusterVariance2,beta))

#Cluster Center Estimate
axs[0,0].plot(x.iloc[:,0], label="Estimate Center 1:%.2f"%x.iloc[-1,0])
axs[0,0].plot(x.iloc[:,1], label="Estimate Center 2: %.2f"%x.iloc[-1,1])
axs[0,0].axhline(y=float(centers.values[0,0]), color="g", label="True Center 1: %.2f"%centers.values[0,0])
axs[0,0].axhline(y=float(centers.values[0,1]), color="r", label="True Center 2: %.2f"%centers.values[0,1])
axs[0,0].title.set_text("Cluster Center Estimates (x_estimate)")
axs[0,0].set_xlabel("Iterations")
axs[0,0].legend()

# Observations 
y = y.values[0]
assignments = assignments.values[0]


n, _, _ = axs[0,1].hist(y,bins=abs(centers.values[0,0]-centers.values[0,1])+10,linewidth=5)
# Distributions
margin = 4
xs1 = np.linspace(centers.values[0,0]-clusterVariance1*margin, centers.values[0,0]+clusterVariance1*margin,100)
xs2 = np.linspace(centers.values[0,1]-clusterVariance2*margin, centers.values[0,1]+clusterVariance2*margin,100)
d1 = stats.norm.pdf(xs1, centers.values[0,0], clusterVariance1) 
d2 = stats.norm.pdf(xs2, centers.values[0,1], clusterVariance2) 
axs[0,1].plot(xs1,d1*n.max()/d1.max(),color="r")
axs[0,1].plot(xs2,d2*n.max()/d2.max(),color="g")
axs[0,1].set_xlabel("Value")
axs[0,1].title.set_text("Observations")

#Cost Functions 

axs[0,2].plot(costX.values[:,0], label=priorX)
axs[0,2].set_xlabel("Iterations")
axs[0,2].title.set_text("Cost X")
axs[0,2].legend()

axs[1,2].plot(costZ.values[:,0], label=priorZ)
axs[1,2].set_xlabel("Iterations")
axs[1,2].title.set_text("Cost Z")
axs[1,2].legend()

#Plot variances 
axs[1,0].plot(s_x.iloc[:,0], label="Variance s_x0")
axs[1,0].plot(s_x.iloc[:,1], label="Variance s_x1")
axs[1,0].title.set_text("Variance S_xi")
axs[1,0].set_xlabel("Iteration")
axs[1,0].legend()

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




# scatterPlot = axs[1,1].scatter(s_z_it, s_z_estimate,c=s_z_assignment,cmap='jet')
axs[1,1].hist(s_z_estimate, bins=1000)
# axs[3].legend(scatterPlot.legend_elements()[0],["0: %i"%counts[0],"1: %i"%counts[1],"Unassigned: %i"%counts[numberClusters]])
axs[1,1].title.set_text("Non-zero S_z values (No assignment for "+str(len(unassigned))+" samples)")
axs[1,1].set_ylabel("Value of S_Zi")
axs[1,1].set_xlabel("Observation i")

fig.set_size_inches((11, 8.5), forward=False)
fig.savefig("./result.png", dpi=500)
# plt.show()


