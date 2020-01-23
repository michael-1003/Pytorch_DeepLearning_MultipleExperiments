#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import csv


#%%
ROOT_DIR = './exp_results/exp2'


#%%
header = open(os.path.join(ROOT_DIR,'exp2_result.csv')).readline()[:-1].split(',')
result = np.genfromtxt(os.path.join(ROOT_DIR,'exp2_result.csv'),delimiter=',',skip_header=1)

def show_paramter(result, header, index):
    line = result[index,:]
    params = {
        header[3]:int(line[3]),
        header[4]:int(line[4]),
        header[6]:line[6],
        header[7]:int(line[7]),
        header[8]:line[8],
        header[9]:line[9],
        header[13]:line[13]
    }
    return params


#%%
# 1. Overview
acc_all = result[:,13]
plt.figure()
x = plt.hist(acc_all,bins=np.arange(0,1.1,0.1))
plt.xticks(np.arange(0,1.1,0.1))
plt.grid()
plt.title('Overview of results')

best_acc_idx = acc_all.argmax()
print('Best parameter:')
print(show_paramter(result,header,best_acc_idx))

worst_acc_idx = acc_all.argmin()
print('Worst parameter:')
print(show_paramter(result,header,worst_acc_idx))


#%%
# 2. Effect of learning rate

