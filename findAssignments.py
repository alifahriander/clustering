import os 
import argparse
import scipy.stats as stats
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt 
import numpy as np 
from collections import Counter
from matplotlib.pyplot import rcParams

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str)
args = parser.parse_args()
PATH = args.path
DIRPATH = os.path.dirname(PATH)
# Read data 
y = pd.read_csv(os.path.join(PATH,"y_observed.csv"), header=None)
y = np.array(y.values[0])
overlapRegion = y [(y[:]>=-1)&(y[:]<=1)]
print(overlapRegion)
print(len(overlapRegion))

assignments = pd.read_csv(os.path.join(PATH,"assignments.csv"), header=None)
assignments = assignments.values[0]

s_z = pd.read_csv(os.path.join(PATH,"s_z.csv"), header=None)
s_z = s_z.values[-1]

z = pd.read_csv(os.path.join(PATH,"z.csv"), header=None)
z = z.values[-1]

zVariances = [list(s_z[0::2]) , list(s_z[1::2])]
# # zVariances = [list(z[0::2]) , list(z[1::2])]
def findMinVariance(variances):
    length = len(variances[0])
    estimateAssignments = []
    for i in range(length):
        if(abs(variances[0][i]) < abs(variances[1][i])):
            estimateAssignments.append(0)
        else:
            estimateAssignments.append(1)
    return estimateAssignments

estimateAssignments = findMinVariance(zVariances)

mistakes = abs(assignments - estimateAssignments)
print(mistakes)
index = np.where(mistakes==1)
print(index)
print(y[index[0]])
print(len(y[index[0]]))