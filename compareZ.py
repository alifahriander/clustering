import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import glob

def plot(path):
    NR_CLUSTERS = 2 
    NR_SHOW = 5
    #TODO: Change scales for assignments add s_z visualizing
    nameDict = {"r":"Red", "b":"Blue"}

    z = pd.read_csv(os.path.join(path,"z.csv"), header=None)
    y = pd.read_csv(os.path.join(path,"y_observed.csv"), header=None)
    y = np.array(y.iloc[0,:])

    trueAssignments = pd.read_csv(os.path.join(path,"assignments.csv"),header=None)
    trueAssignments = np.array(trueAssignments.iloc[0,:])

    x_estimate = pd.read_csv(os.path.join(path,"x_estimate.csv"), header=None)
    x_estimate = np.array(x_estimate.iloc[-1,:])

    x_true = pd.read_csv(os.path.join(path,"x_true.csv"), header=None)
    x_true = np.array(x_true.iloc[-1,:])

    # Determine colors
    TRUE1_COLOR = 'r'
    TRUE2_COLOR = 'b'
    TRUE1_NAME = "1"
    TRUE2_NAME = "2"

    ESTIMATE1_COLOR = None
    ESTIMATE2_COLOR = None
    ESTIMATE1_NAME = None
    ESTIMATE2_NAME = None

    if(np.linalg.norm(x_true-x_estimate) < np.linalg.norm(x_true-np.flip(x_estimate))):
        ESTIMATE1_COLOR = TRUE1_COLOR
        ESTIMATE2_COLOR = TRUE2_COLOR

        ESTIMATE1_NAME = TRUE1_NAME
        ESTIMATE2_NAME = TRUE2_NAME

    else:
        ESTIMATE1_COLOR = TRUE2_COLOR
        ESTIMATE2_COLOR = TRUE1_COLOR

        ESTIMATE1_NAME = TRUE2_NAME
        ESTIMATE2_NAME = TRUE1_NAME
        print("Flipped")



    s_z = pd.read_csv(os.path.join(path,"s_z.csv"), header=None)
    lastSz = np.array(s_z.iloc[-1,:].values)
    sz1 = lastSz[0::2]
    sz2 = lastSz[1::2]

    last = z.iloc[-1,:]
    last = np.array(last)

    estimate = []

    for i in range(last.shape[0]//NR_CLUSTERS):
        if(abs(last[NR_CLUSTERS*i]) < abs(last[NR_CLUSTERS*i+1])):
            estimate.append(0)
        else:
            estimate.append(1)
    estimate = np.array(estimate)

    last = list(np.abs(last))

    # Color selection
    cmap = []
    for i in range(NR_SHOW):
        if(trueAssignments[i]==0):
            cmap.append(TRUE1_COLOR)
        else:
            cmap.append(TRUE2_COLOR)

    fig, axs = plt.subplots(2,3)

    fig.suptitle("Cluster 1 color: %s , Cluster 2 Color: %s" % (nameDict[ESTIMATE1_COLOR], nameDict[ESTIMATE2_COLOR]))


    axs[0,0].scatter(list(range(NR_SHOW)),y[:NR_SHOW],c=cmap)
    axs[0,0].title.set_text("Y")
    axs[0,0].set_xlabel("Index")
    axs[0,0].set_ylabel("Value")
    axs[0,0].xaxis.get_major_locator().set_params(integer=True)

    axs[0,1].plot(last[0::2][:NR_SHOW],ESTIMATE1_COLOR+'o')
    axs[0,1].title.set_text("Z_"+ESTIMATE1_NAME)
    axs[0,1].set_xlabel("Index")
    axs[0,1].set_ylabel("Value")
    axs[0,1].xaxis.get_major_locator().set_params(integer=True)


    axs[1,1].plot(sz1[:NR_SHOW],ESTIMATE1_COLOR+'*')
    axs[1,1].title.set_text("SZ_"+ESTIMATE1_NAME)
    axs[1,1].set_xlabel("Index")
    axs[1,1].set_ylabel("Value")
    axs[1,1].xaxis.get_major_locator().set_params(integer=True)

    axs[0,2].plot(last[1::2][:NR_SHOW],ESTIMATE2_COLOR+'o')
    axs[0,2].title.set_text("Z_"+ESTIMATE2_NAME)
    axs[0,2].set_xlabel("Index")
    axs[0,2].set_ylabel("Value")
    axs[0,2].xaxis.get_major_locator().set_params(integer=True)

    axs[1,2].plot(sz2[:NR_SHOW],ESTIMATE2_COLOR+'*')
    axs[1,2].title.set_text("SZ_"+ESTIMATE2_NAME)
    axs[1,2].set_xlabel("Index")
    axs[1,2].set_ylabel("Value")
    axs[1,2].xaxis.get_major_locator().set_params(integer=True)

    axs[1,0].axhline(y=x_true[0], color=TRUE1_COLOR, label="True Cluster"+TRUE1_NAME)
    axs[1,0].axhline(y=x_true[1], color=TRUE2_COLOR, label="True Cluster"+TRUE2_NAME)
    axs[1,0].scatter([0],x_estimate[0],c=[ESTIMATE1_COLOR],label="Estimate Cluster"+ESTIMATE1_NAME)
    axs[1,0].scatter([0],x_estimate[1],c=[ESTIMATE2_COLOR],label="Estimate Cluster"+ESTIMATE2_NAME)
    axs[1,0].title.set_text("X Estimate")
    axs[1,0].legend()
    axs[1,0].axes.get_xaxis().set_visible(False)


    plt.subplots_adjust(wspace=0.34,hspace=0.31)
    
    fig.set_size_inches((11, 8.5), forward=False)
    fig.savefig(os.path.join("zImages",path.replace("/","-")+"z.png"), dpi=500)

if __name__ == "__main__":
    path = "experiments"
    folders = glob.glob(os.path.join(path,"*/*/"))
    folders = sorted(folders)
    for f in folders:
        plot(f)
