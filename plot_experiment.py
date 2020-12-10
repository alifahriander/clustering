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
s_z = pd.read_csv(os.path.join(PATH,"s_z.csv"), header=None)
y = pd.read_csv(os.path.join(PATH,"y_observed.csv"), header=None)
costZ = pd.read_csv(os.path.join(PATH,"costZ.csv"), header=None)

r_z = pd.read_csv(os.path.join(PATH,"config.csv"), header=None, index_col=0).loc["r_z"][1]
centers = pd.read_csv("experiments/config/centers.csv", header=None).loc[:,0].values
variances = pd.read_csv("experiments/config/variances.csv", header=None).loc[:,0].values

maximumCenter = np.max(centers)
minimumCenter = np.min(centers)

fig, axs = plt.subplots(1,3)
# axs = axs.ravel()
fig.suptitle("Experiment: r_z %.4f"%(r_z))

#Cluster Center Estimate
for i in range(numberClusters):
    axs[0].plot(x.iloc[:,i], label="Estimate Center %i:%.2f"%(i,x.iloc[-1,i]))
    axs[0].axhline(y=float(centers[i]), color ='k', label="True Center %i: %.2f"%(i,centers[i]))
axs[0].title.set_text("Cluster Center Estimates (x_estimate)")
axs[0].set_xlabel("Iterations")

# # Observations 
y = y.values[0]
n, _, _ = axs[1].hist(y, bins=100,linewidth=5)
# Distributions
margin = 4
for i in range(numberClusters):
    xs= np.linspace(centers[i]-variances[i]*margin, centers[i]+variances[i]*margin,100)
    distrb = stats.norm.pdf(xs, centers[i], variances[i]) 
    axs[1].plot(xs,distrb*n.max()/distrb.max())
axs[1].set_xlabel("Value")
axs[1].title.set_text("Observations")

#Cost Functions 
axs[2].plot(costZ.values[:,0])
axs[2].set_xlabel("Iterations")
axs[2].title.set_text("Cost Z")

plt.subplots_adjust(bottom=0.25,top=0.75)

fig.set_size_inches((11, 8.5), forward=False)
fig.savefig("./result.png", dpi=500)
# plt.show()


