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

scores = df["score"].values
x = np.log10(df["r_z"].values)
plt.scatter(x,scores,c='b',label="Score")

stats = df.groupby(['experiment','r_z']).agg({'score': ['mean','min', 'max']}).sort_values(by=("score","mean"),ascending=True)
print(os.path.join(os.path.dirname(path_csv),"stats.csv"))
stats.to_csv(os.path.join(os.path.dirname(path_csv),"stats.csv"))



# rzs =stats.index.get_level_values("r_z").values
# plt.scatter(np.log10(rzs), stats.iloc[:,0].values,c='r',label="Mean")
# plt.title("Scores for R_z")
# plt.xlabel("R_z (10^i)")
# plt.ylabel("Score")
# plt.legend()

# plt.show()