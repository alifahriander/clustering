import pandas as pd 
import numpy as np 
import os
import glob
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str,required=True)
args = parser.parse_args()
PATH = args.path

dirs = sorted(glob.glob(os.path.join(PATH,"experiment*/*/x_estimate.csv")))
dirs = [ os.path.dirname(x) for x in dirs ]
name_x_estimate = "x_estimate.csv"
name_x_true = "x_true.csv"
name_config = "config.csv"

path_csv = "./results_all/result.csv"

path_config = os.path.join(dirs[0],name_config)
config = pd.read_csv(path_config,header=None)
config = config.transpose()
config.columns = config.iloc[0]
df = pd.DataFrame(columns=["score","folder","experiment",*config.columns])

for d in dirs:
    path_x_estimate = os.path.join(d,name_x_estimate)
    path_x_true = os.path.join(d,name_x_true)
    path_config = os.path.join(d,name_config)


    x_estimate = pd.read_csv(path_x_estimate,header=None)
    x_true = pd.read_csv(path_x_true,header=None)
    config = pd.read_csv(path_config,header=None)

    v_estimate = x_estimate.iloc[-1,:].values
    v_true = x_true.iloc[0,:].values

    d1 = np.linalg.norm(v_true-v_estimate)
    d2 = np.linalg.norm(v_true-np.flip(v_estimate))

    score = d1 if d1<d2 else d2
    config = config.transpose()
    config.columns = config.iloc[0]
    config = config.drop(config.index[0])
    config["score"] = score 
    config["folder"] = d
    config["experiment"] = os.path.dirname(d)
    config["numberOfIterations"] = x_estimate.index[-1]
    df = df.append(dict(zip(config.columns,config.values[0])),ignore_index=True)

df = df.sort_values(by=["score"],ascending=True)
df.to_csv(path_csv,index=False)
print(df)

