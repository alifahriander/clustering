import pandas as pd 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os 
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str,required=True)
args = parser.parse_args()
path_csv = args.path



df = pd.read_csv(path_csv)

stats = df.groupby(['experiment','beta','r_x','r_z']).agg({'score': ['mean','min', 'max']}).sort_values(by=("score","min"),ascending=True)
print(stats.iloc[1,:])
stats.to_csv("./results_all/stats.csv")
rxs = stats.index.get_level_values("r_x").values
rzs =stats.index.get_level_values("r_z").values
betas = stats.index.get_level_values("beta").values
 

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.set_xlabel("r_x")
ax.set_ylabel("beta")
ax.set_zlabel("r_z")
ax.set_facecolor('black')

ax.xaxis.label.set_color('red')
ax.tick_params(axis='x', colors='white')

ax.yaxis.label.set_color('red')
ax.tick_params(axis='y', colors='white')

ax.zaxis.label.set_color('red')
ax.tick_params(axis='z', colors='white')



ax.title.set_text("Minimal Hyperparameter Scores in 10^i")
img = ax.scatter(np.log10(rxs), np.log10(betas), np.log10(rzs), c=stats.iloc[:,1].values, cmap=plt.jet())





fig.colorbar(img)
plt.show()

