import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
path = "experiments_MediumVariance/experiment3/run2"

NR_CLUSTERS = 2 
NR_SHOW = 20
#TODO: Change scales for assignments add s_z visualizing


z = pd.read_csv(os.path.join(path,"z.csv"), header=None)
trueAssignments = pd.read_csv(os.path.join(path,"assignments.csv"),header=None)
trueAssignments = np.array(trueAssignments.iloc[0,:])

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

fig, axs = plt.subplots(5,1)

axs[0].plot(trueAssignments[:NR_SHOW],'gx')
axs[0].title.set_text("True Assignments")
axs[1].plot(last[1::2][:NR_SHOW],'ro')
axs[1].title.set_text("Z_1")

axs[1].plot(sz1[1::2][:NR_SHOW],'r*')
axs[1].title.set_text("SZ_1")

axs[3].plot(last[0::2][:NR_SHOW],'bo')
axs[3].title.set_text("Z_2")

axs[4]

plt.show()