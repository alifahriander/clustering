import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 

df = pd.read_csv("./experiments/experiment7/run2/s_z.csv", header=None)

matrix = df.values 
matrix_mean = np.mean(matrix, axis=1)
matrix_std = np.std(matrix, axis=1)

fig, axs = plt.subplots(1,2)

axs[0].plot(matrix_mean)
axs[0].title.set_text("Mean")
axs[1].plot(matrix_std)
axs[1].title.set_text("Standart Deviation")

plt.show()