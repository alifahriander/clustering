import pandas as pd 
import numpy as np 
import os
import glob
import argparse
from scipy.spatial.distance import cdist
def computeDistances(vTrue, vEstimate):
    distanceMatrix = []
    for i in vTrue:
        row = []
        for j in vEstimate:
            row.append(np.linalg.norm(i-j,))
        distanceMatrix.append(row)
    return np.array(distanceMatrix)

def computeMinCost(matrix):
    distanceMatrix = matrix.copy()
    cost = 0.0
    # for row in distanceMatrix:
    while(np.amin(distanceMatrix) != np.inf):
        idx = np.unravel_index(np.argmin(distanceMatrix, axis=None), distanceMatrix.shape)
        cost += distanceMatrix[idx]
        distanceMatrix[idx[0],:] = np.inf
        distanceMatrix[:,idx[1]] = np.inf
    return cost





parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str,required=True)
args = parser.parse_args()
PATH = args.path

dirs = sorted(glob.glob(os.path.join(PATH,"experiment*/*/x_estimate.csv")))
dirs = [ os.path.dirname(x) for x in dirs ]
name_x_estimate = "x_estimate.csv"
name_x_true = "x_true.csv"
name_config = "config.csv"

path_csv = os.path.join(PATH,"result.csv")

df = pd.DataFrame(columns=["experiment","folder","r_z","score",])

for d in dirs:
    path_x_estimate = os.path.join(d,name_x_estimate)
    path_x_true = os.path.join(d,name_x_true)
    path_config = os.path.join(d,name_config)


    x_estimate = pd.read_csv(path_x_estimate,header=None)
    x_true = pd.read_csv(path_x_true,header=None)
    r_z = pd.read_csv(path_config,header=None, index_col=0).loc["r_z",1]


    v_estimate = np.array([[x] for x in x_estimate.iloc[-1,:]])
    v_true = np.array([[x] for x in x_true.iloc[-1,:]])

    distances = computeDistances(v_true, v_estimate)
    score = computeMinCost(distances)

    df = df.append({"experiment":os.path.dirname(d),"folder":d,"r_z":r_z,"score":score},ignore_index=True)

df = df.sort_values(by=["score"],ascending=True)
df.to_csv(path_csv,index=False)
print(df)

