import os
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

EXP_RES_DIR = './'
EXP_NAME = 'exp1'
NUM_EXP = 6
LABELS = ['vgg16', 'vgg13', 'resnet18', 'resnet34', 'senet18', 'densenet']

train_losses = []
valid_losses = []
valid_scores = []
for i in range(NUM_EXP):
    case_dir = EXP_RES_DIR + str(i) +'/'
    train_loss = np.genfromtxt((case_dir+'train_losses.txt'),delimiter=',')
    valid_loss = np.genfromtxt((case_dir+'valid_losses.txt'),delimiter=',')
    valid_score = np.genfromtxt((case_dir+'valid_scores.txt'),delimiter=',')

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    valid_scores.append(valid_score)

train_losses = np.dstack(train_losses)
valid_losses = np.dstack(valid_losses)
valid_scores = np.dstack(valid_scores)

title = 'train_losses'
plt.figure(figsize=(6,4))
plt.plot(train_losses[:,0,:],train_losses[:,1,:])
plt.legend(LABELS)
plt.title(EXP_NAME + ': ' + title)
plt.grid()
plt.savefig((title+'.png'), dpi=300)
plt.close('all')

title = 'valid_losses'
plt.figure(figsize=(6,4))
plt.plot(valid_losses[:,0,:],valid_losses[:,1,:])
plt.legend(LABELS)
plt.title(EXP_NAME + ': ' + title)
plt.grid()
plt.savefig((title+'.png'), dpi=300)
plt.close('all')

title = 'valid_scores'
plt.figure(figsize=(6,4))
plt.plot(valid_scores[:,0,:],valid_scores[:,1,:])
plt.legend(LABELS)
plt.title(EXP_NAME + ': ' + title)
plt.grid()
plt.savefig((title+'.png'), dpi=300)
plt.close('all')

################################################################
test_accs = np.genfromtxt('test_scores.txt',delimiter=',')

cmap = cm.get_cmap('tab10')
color = [cmap(i) for i in range(NUM_EXP)]

title = 'test_accuarcy'
fig, ax = plt.subplots(figsize=(6,4))
rects = ax.bar(LABELS, test_accs[:,1], width=0.5, bottom=0, color=color)
plt.grid()
plt.ylim([0.8,1])
ax.set_title(EXP_NAME + ': ' + title)
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.4f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
autolabel(rects)
plt.savefig((title+'.png'), dpi=300)
plt.close('all')


