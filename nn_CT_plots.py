# Plot loss results from digit_recog_CNN.py
import tensorflow as tf
import random as ran
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import argparse
import time
import pickle

#############################################
# HYPER-PARAMETERS
#############################################
parser = argparse.ArgumentParser()

# General
parser.add_argument('--max_steps', type=int, default='1000',
                    help='Maximum number of steps.')
parser.add_argument('--version', type=str, default='original',
                    help='Version name.')
parser.add_argument('--lr', type=float, default='0.5e-3',
                    help='Learning rate.')
parser.add_argument('--vali', type=float, default='0.1',
                    help='Size of validation set relative to training set.')
parser.add_argument('--n_fc', type=int, default='2',
                    help='Number of fully-connected layers.')
                    # The final output layer is not considered in n_fc

FLAGS = parser.parse_args()

max_steps = FLAGS.max_steps
eta = FLAGS.lr
version = FLAGS.version
vali = FLAGS.vali
n_fc = FLAGS.n_fc

#############################################
# LOSS AND PREDICTION ERROR
#############################################
crs = open("logs_loss/loss_%s_%s_%s_%s_%s.out" % (eta,n_fc,version,vali,max_steps), "r")

step = []
loss_train = []
loss_validation = []
train_acc = []
validation_acc = []

k = 0
for line in crs:
    if k>2:
        step.append(line.split()[0])
        loss_train.append(line.split()[1])
        loss_validation.append(line.split()[3])
        train_acc.append(line.split()[2])
        validation_acc.append(line.split()[4])
    k = k+1

crs.close()

# Minimum value of loss_validation:
min_loss = min(loss_validation)
min_index = loss_validation.index(min(loss_validation))
min_step = step[min_index]
print('*** Minimum loss: %.3f; reached at step %d ***'%(float(min_loss),int(min_step)))

# Plots
plt.figure(1)
plt.plot(step,loss_train,label="Training")
plt.plot(step,loss_validation,label="Validation")
plt.ylabel('Loss')
plt.xlabel('Step')
plt.legend(loc=1)
#plt.xscale('log')
plt.yscale('log')
plt.grid(True,which='minor')
plt.show()

plt.figure(2)
plt.plot(step,train_acc,label="Training")
plt.plot(step,validation_acc,label="Validation")
plt.ylabel('Average prediction error per batch (%)')
plt.xlabel('Step')
plt.legend(loc=4)
plt.grid(True)
plt.show()
