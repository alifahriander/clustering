import os 
import argparse
import scipy.stats as stats
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt 
import numpy as np 
from collections import Counter
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 4.7, 4
# Save as png for latex 
# matplotlib.use("png")
# matplotlib.rcParams.update({
#     "png.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'png.rcfonts': False,
# })

# COLORS
BLUE = "#0000FF"
RED = "#FF0000"
GREEN = "#00FF00"
BLACK = "#000000"

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str)
args = parser.parse_args()
PATH = args.path
DIRPATH = os.path.dirname(PATH)
# Read data 
centers = pd.read_csv(os.path.join(PATH,"x_true.csv"), header=None)
numberClusters = len(centers.values[0])
print("numberClusters:", numberClusters)

x = pd.read_csv(os.path.join(PATH,"x_estimate.csv"), header=None)
x = x.values[0]

y = pd.read_csv(os.path.join(PATH,"y_observed.csv"), header=None)
y = y.values[0]

assignments = pd.read_csv(os.path.join(PATH,"assignments.csv"), header=None)
assignments = assignments.values[0]

z = pd.read_csv(os.path.join(PATH,"z.csv"), header=None)

s_x = pd.read_csv(os.path.join(PATH,"s_x.csv"), header=None)
s_z = pd.read_csv(os.path.join(PATH,"s_z.csv"), header=None)

r_z = pd.read_csv(os.path.join(PATH,"config.csv"), header=None, index_col=0).loc["r_z"][1]

# Convert assignments to colors
def colorMapping(assignment):
    if(assignment==0):
        return BLUE
    elif(assignment==1):
        return RED
    elif(assignment==2):
        return GREEN

assignments = list(assignments)
assignments = list(map(colorMapping, assignments))


# Split s_z for all clusters
s_z = s_z.values[-1]
zVariances = [list(s_z[0::2]) , list(s_z[1::2])]

# Split z for all clusters 
z = z.values[-1]
zSplit = [list(z[0::2]), list(z[1::2])]

# Plot Histogram of Data and Estimates 
n,_,_ = plt.hist(y, bins=100, linewidth=5, color=BLACK)
plt.xlabel("Location")
plt.ylabel("Number of Units")
centers = [-2,2]
variances = [1.0, 1.0]
margin = 4
numberClusters = 2
colors = [BLUE, RED]
for i in range(numberClusters):
    xs= np.linspace(centers[i]-variances[i]*margin, centers[i]+variances[i]*margin,100)
    distrb = stats.norm.pdf(xs, centers[i], variances[i]) 
    plt.plot(xs,distrb*n.max()/distrb.max(), color=colors[i])
# plt.show()
plt.savefig(os.path.join(PATH,"y.png"))
plt.clf()

# # Plot Variances 
plt.bar(range(0,len(s_z)//2),zVariances[0], color=assignments)
plt.xlabel("Sample Index")
plt.ylabel("Value")
# plt.show()
plt.savefig(os.path.join(PATH,"sz1.png"))
plt.clf()

plt.bar(range(0,len(s_z)//2),zVariances[1], color = assignments)
plt.xlabel("Sample Index")
plt.ylabel("Value")
# plt.show()
plt.savefig(os.path.join(PATH,"sz2.png"))
plt.clf()

# Plot Z 
plt.bar(range(0,len(zSplit[0])), zSplit[0], color=assignments)
plt.xlabel("Sample Index")
plt.ylabel("Value")
# plt.show()
plt.savefig(os.path.join(PATH,"z1.png"))
plt.clf()

plt.bar(range(0,len(zSplit[1])), zSplit[1], color=assignments)
plt.xlabel("Sample Index")
plt.ylabel("Value")
plt.savefig(os.path.join(PATH,"z2.png"))
# plt.show()
plt.clf()
