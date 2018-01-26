import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvnorm

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import tkinter as tk

# Read the Data from the CVS file
df = pd.DataFrame.from_csv("1challengeSt.csv")
df_L1 = df.loc[df['label'] == 1.0]
df_L0 = df.loc[df['label'] == 0.0]
dftest = df.loc[~((df['label'] == 0.0) | (df['label'] == 1.0))]

# Cast it to numpy arrays
TrainingData_L1 = df_L1.as_matrix(columns=None)
TrainingData_L0 = df_L0.as_matrix(columns=None)
TestData = dftest.as_matrix(columns=['Y0', 'Y1'])

# Fit a Normal distribution to each training data set
UseTheGuess = True
if not UseTheGuess:
    mean_L1 = np.mean(TrainingData_L1[:, 0:2], axis=0)  # np.asarray([0.0,0.0])
    cov_L1 = np.cov(TrainingData_L1[:, 0:2], rowvar=0)  # np.asarray([[1.0,0.7],[0.7,1.0]])
    mean_L0 = np.mean(TrainingData_L0[:, 0:2], axis=0)  # np.asarray([0.0,0.0])
    cov_L0 = np.cov(TrainingData_L0[:, 0:2], rowvar=0)  # np.asarray([[1.0,0],[0,1.0]])
else:
    mean_L1 = np.asarray([0.0, 0.0])
    cov_L1 = np.asarray([[1.0, 0.7], [0.7, 1.0]])
    mean_L0 = np.asarray([0.0, 0.0])
    cov_L0 = np.asarray([[1.0, 0], [0, 1.0]])

# Evaluate and compare the likelihoods and generate the labels
L0 = mvnorm.pdf(TestData, mean=mean_L0, cov=cov_L0)
L1 = mvnorm.pdf(TestData, mean=mean_L1, cov=cov_L1)
LikelihoodRatio = np.divide(L1, L0)
tau = 1.3805  # This value minimizes the error functions defined at the very end of the file # TrainingData_L0.shape[0]/TrainingData_L1.shape[0]  # LikelihoodRatio>tau-> 1
dftest.values[:, 2] = 0
dftest.values[LikelihoodRatio > tau, 2] = 1

# update the original CSV file
df = pd.concat([df_L0, df_L1, dftest], join='outer', ignore_index=True)
df.to_csv("1challenge.csv")

# Some plots to see what's going on
plt.figure(0)
f0, ax0 = plt.subplots(ncols=2)
f0.set_size_inches(11, 5)
plt.subplots_adjust(left=0.04, right=0.96, bottom=0.06, top=0.94, wspace=0.1)
ax0[0].set_xlim([-7, 7])
ax0[0].set_ylim([-7, 7])
ax0[0].axis('equal')
ax0[1].axis('equal')
ax0[1].plot(TrainingData_L1[:, 0], TrainingData_L1[:, 1], 'x', color='r')
ax0[0].plot(TrainingData_L0[:, 0], TrainingData_L0[:, 1], 'x', color='b')
dftest0 = dftest.loc[df['label'] == 0.0]
dftest1 = dftest.loc[df['label'] == 1.0]
ax0[1].plot(dftest1.values[:, 0], dftest1.values[:, 1], 'o', color='k')
ax0[0].plot(dftest0.values[:, 0], dftest0.values[:, 1], 'o', color='k')
f0.suptitle('The DF results saved to cvs')
f0.show()

plt.figure(1)
f, ax = plt.subplots(ncols=2)
f.set_size_inches(11, 5)
plt.subplots_adjust(left=0.04, right=0.96, bottom=0.06, top=0.94, wspace=0.1)
ax[0].set_xlim([-7, 7])
ax[0].set_ylim([-7, 7])
ax[0].axis('equal')
ax[1].axis('equal')
ax[1].plot(TrainingData_L1[:, 0], TrainingData_L1[:, 1], 'x', color='r')
ax[0].plot(TrainingData_L0[:, 0], TrainingData_L0[:, 1], 'x', color='b')
init = True


def update_plot(self):
    global ax0TeD, ax1TeD, init
    _tau = float(self)
    TestData_L1 = TestData[LikelihoodRatio > _tau, :]
    TestData_L0 = TestData[LikelihoodRatio <= _tau, :]
    if init:
        ax1TeD, = ax[1].plot(TestData_L1[:, 0], TestData_L1[:, 1], 'o', color='k')
        ax0TeD, = ax[0].plot(TestData_L0[:, 0], TestData_L0[:, 1], 'o', color='k')
        init = False
    else:
        ax1TeD.set_xdata(TestData_L1[:, 0])
        ax1TeD.set_ydata(TestData_L1[:, 1])
        ax0TeD.set_xdata(TestData_L0[:, 0])
        ax0TeD.set_ydata(TestData_L0[:, 1])
    f.canvas.draw_idle()
    e.delete(0, tk.END)
    e.insert(0, _tau)
    # error = (TestData_L0.shape[0] / TestData_L1.shape[0]) - ( (TestData_L0.shape[0]+TrainingData_L0.shape[0]) / (TestData_L1.shape[0]+TrainingData_L1.shape[0]))
    # error = (TestData_L0.shape[0] / TestData_L1.shape[0]) - ( (TrainingData_L0.shape[0]) / (TrainingData_L1.shape[0]))
    error = (TestData_L0.shape[0] / (TestData_L1.shape[0]+TestData_L0.shape[0])) - ((TrainingData_L0.shape[0]) / (TrainingData_L1.shape[0]+TrainingData_L0.shape[0]))
    eText.set('   The Error is: ' + str(error))


root = tk.Tk()
root.wm_title("Interactive Tau")
frame = tk.Frame(root)
frame.pack(side=tk.TOP, fill=tk.X)
canvas = FigureCanvasTkAgg(f, master=root)
canvas.show()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
toolbar = NavigationToolbar2TkAgg(canvas, frame)
w = tk.Scale(master=root, from_=.5, to=2.5, orient=tk.HORIZONTAL, command=update_plot, resolution=0.0001, length=600)
w.set(value=1.3805)
w.pack()
frame2 = tk.Frame(root)
frame2.pack()


def callback():
    update_plot(e.get())
    w.set(e.get())


e = tk.Entry(master=frame2)
e.pack(side=tk.LEFT)
e.insert(0, w.get())
e.bind('<Return>', callback)
b = tk.Button(master=frame2, text="Apply", width=10, command=callback)
b.pack(side=tk.LEFT)
eText = tk.StringVar()
e2 = tk.Entry(master=frame2, width=40, textvariable=eText)
e2.pack(side=tk.LEFT)
eText.set('   The Error is: ')
e2.config(state='readonly')


tk.mainloop()
''' 
All kinda the same and have to be minimized 
error = (TestData_L0.shape[0] / TestData_L1.shape[0]) - ( (TestData_L0.shape[0]+TrainingData_L0.shape[0]) / (TestData_L1.shape[0]+TrainingData_L1.shape[0]))
error = (TestData_L0.shape[0] / TestData_L1.shape[0]) - ( (TrainingData_L0.shape[0]) / (TrainingData_L1.shape[0]))
error = (TestData_L0.shape[0] / (TestData_L1.shape[0]+TestData_L0.shape[0]) ) - ( (TrainingData_L0.shape[0]) / (TrainingData_L1.shape[0]+TrainingData_L0.shape[0]))
'''
