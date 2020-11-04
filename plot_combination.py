import os 
import matplotlib.pyplot as plt 
import argparse
import glob
import pandas as pd 


# Get experiment folder 
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str)
args = parser.parse_args()
PATH = args.path

# Find every x_estimate.csv in this folder
files_x = sorted(glob.glob(os.path.join(PATH,"*/x_estimate.csv")))



x_true = pd.read_csv(os.path.join(PATH,"run1/x_true.csv"),header=None)
ax = plt.gca()
fig, axs = plt.subplots(1,2)
for i in range(1,len(files_x)+1):
    path = os.path.join(PATH,"run"+str(i))
    path_xestimate = os.path.join(path,"x_estimate.csv")
    path_sx = os.path.join(path,"s_x.csv")

    x_estimate = pd.read_csv(path_xestimate, header=None)
    # To get the same color for the same run
    color = next(ax._get_lines.prop_cycler)['color']

    axs[1].scatter([i,i],[*x_estimate.iloc[-1,:]],color=color, marker='o')
    axs[0].scatter([i,i],[*x_estimate.iloc[0,:]],color=color, marker='x')
    # pairs = [(x_estimate.iloc[-1,0],x_estimate[0,0]),(x_estimate.iloc[-1,1], x_estimate.iloc[0,1])]
    # print(i)
    # print(x_estimate.iloc[-1,0])
    # axs[0].plot([i,i],[x_estimate.iloc[-1,0],x_estimate.iloc[0,0]],color = "black",linestyle="dashed")
    # axs[0].plot([i,i],[x_estimate.iloc[-1,1],x_estimate.iloc[0,1]], color="black",linestyle="solid")

    axs[0].title.set_text("First x_estimate locations")
    axs[0].set_xlabel("Run")
    axs[0].set_ylabel("Location")

    axs[1].title.set_text("Last x_estimate locations")
    axs[1].set_xlabel("Run")
    axs[1].set_ylabel("Location")

    # axs[1].plot(x_estimate.iloc[:,0],color=color)
    # axs[1].plot(x_estimate.iloc[:,1],color=color)
    # axs[1].set_xlabel("Iteration")
    # axs[1].set_ylabel("X_estimate value")
    # axs[1].title.set_text("X_estimate for every run")


    # s_x = pd.read_csv(path_sx, header=None)
    # # To get the same color for the same run
    # axs[2].plot(s_x.iloc[:,0],color=color)
    # axs[2].plot(s_x.iloc[:,1],color=color)
    # axs[2].set_xlabel("Iteration")
    # axs[2].set_ylabel("S_x value")
    # axs[2].title.set_text("S_x for every run")



axs[0].axhline(y=float(x_true.values[0,0]), color="r",linewidth=3)
axs[0].axhline(y=float(x_true.values[0,1]), color="r", label="True Centers", linewidth=3)


axs[1].axhline(y=float(x_true.values[0,0]), color="r",linewidth=3)
axs[1].axhline(y=float(x_true.values[0,1]), color="r", label="True Centers", linewidth=3)

plt.legend()
plt.show()