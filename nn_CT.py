##########################################################
# Neural network used for the analysis of the features
# from the CT scan
##########################################################
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
parser.add_argument('--lr', type=float, default='0.5e-10',
                    help='Learning rate.')
parser.add_argument('--vali', type=float, default='0.1',
                    help='Size of validation set relative to training set.')
parser.add_argument('--test', type=float, default='0.05',
                    help='Size of test set relative to training set.')
parser.add_argument('--n_fc', type=int, default='2',
                    help='Number of fully-connected layers.')
                    # The final output layer is not considered in n_fc

# Logs:
parser.add_argument('--log_freq', type=int, default='50',
                    help='Log frequency.')

FLAGS = parser.parse_args()

max_steps = FLAGS.max_steps
eta = FLAGS.lr
version = FLAGS.version
vali = FLAGS.vali
tst = FLAGS.test
n_fc = FLAGS.n_fc

train_batch_size = 32
vali_batch_size = 256

train_restored_model = 'no'
extra_steps = 10000

log_frequency = FLAGS.log_freq
seed = 2
tf.set_random_seed(seed)

#############################################
# READ DATA
#############################################
with open('data/YXnorm_NO_IMG.pickle','rb') as f:  # Python 3: open(..., 'rb')
    Y,X,Yshift,Xshift,Yscale,Xscale = pickle.load(f)

features = np.array(X)
labels = np.array(Y)
labels = np.float32(labels)

assert features.shape[0] == labels.shape[0]

# Shuffle data
num_data = len(labels)
order = np.random.permutation(num_data)
features = features[order, ...]
labels = labels[order]

# Separate training and validation sets
n_vali = int(vali*len(labels))
n_test = int(tst*len(labels))
features_vali = features[0:n_vali]
features_test = features[n_vali:(n_vali+n_test)]
features = features[(n_vali+n_test):]
labels_vali = labels[0:n_vali]
labels_test = labels[n_vali:(n_vali+n_test)]
labels = labels[(n_vali+n_test):]

p = features.shape[1]

#############################################
# PREVIOUS DEFINITIONS
#############################################
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# Create batches by shuffling the data and taking the first 'batch_size' points
def create_batch(features,labels,batch_size):
    # Shuffle data
    num_data = len(labels)
    order = np.random.permutation(num_data)
    features = features[order, ...]
    labels = labels[order]
    return features[0:batch_size], labels[0:batch_size]

def display_filters(weights):
    N0 = tf.cast(weights.shape[3], tf.float32)
    N = sess.run(tf.to_int32(tf.ceil(tf.sqrt(N0))))
    print('Total filters order: %dx%d' % (N,N))
    f, axarr = plt.subplots(N, N)

    p = 0
    for i in range(N):
        for j in range(N):

            # empty plot white when out of kernels to display
            if p >= weights.shape[3]:
                krnl = np.ones((weights.shape[0],weights.shape[1],3))
            else:
                if weights.shape[2] == 8:
                    krnl = weights[:,:,:,p]
                    axarr[i,j].imshow(krnl)
                else:
                    krnl = sess.run(tf.reduce_mean(weights[:,:,:,p],axis=2))
                    axarr[i,j].imshow(krnl,cmap='Reds')
            axarr[i,j].axis('off')
            p+=1

#############################################
# NN
#############################################
# Input layer:
x  = tf.placeholder(tf.float32, [None, p], name='x')

# FC layers:
layers = [x]
layer_i_size = p
keep_prob = tf.placeholder(tf.float32)
for ii in range(0,n_fc):
    with tf.variable_scope('fc_%d' %ii):
        foo = 'W_fc_%d'%ii
        exec(foo + " = weight_variable([layer_i_size, layer_i_size//2])")
        #W = weight_variable([layer_i_size, layer_i_size//2])
        b = bias_variable([layer_i_size//2])
        # Dropout to reduce overfitting:
        #fclayer = tf.nn.dropout(tf.nn.tanh(tf.matmul(layers[-1], W) + b), keep_prob)
        exec("fclayer = tf.nn.dropout(tf.nn.tanh(tf.matmul(layers[-1], "+foo+") + b), keep_prob)")
        layers.append(fclayer)
        layer_i_size = layer_i_size//2

W_out = weight_variable([layer_i_size, 1])
b_out = bias_variable([1])

# The output layer has a sigmoid, to get values from 0 to 1
y_ = tf.nn.sigmoid(tf.matmul(layers[-1],W_out) + b_out)

#############################################
# COST FUNCTION (CROSS-ENTROPY)
#############################################
y = tf.placeholder(tf.float32, [None, 1], name='y')

loss = tf.losses.mean_squared_error(labels=y, predictions=y_)

#############################################
# TRAINING
#############################################
train_step = tf.train.AdamOptimizer(eta).minimize(loss)

#############################################
# EVALUATE AVERAGE PREDICTION ERROR PER BATCH
#############################################
pred_error = tf.reduce_mean(tf.abs(y-y_))

#############################################
# RUN MODEL
#############################################
with tf.Session() as sess:
  # Save or restore model
  saver = tf.train.Saver()
  my_file = Path("nn_models/nn_model_%s_%s_%s_%s_%s.meta.meta" % (eta,n_fc,version,vali,max_steps))
  if my_file.is_file():
      # file exists
      saver.restore(sess, "nn_models/nn_model_%s_%s_%s_%s_%s.meta" % (eta,n_fc,version,vali,max_steps))
      print("Model restored.")
      if train_restored_model in ['y', 'Y', 'yes', 'Yes', 'YES']:
          print("Further training not implemented.")
      else:
          print("Old model in use.")

  else:
      file = open("logs_loss/logs_%s_%s_%s_%s_%s.out" % (eta,n_fc,version,vali,max_steps),"a")
      file.write("INPUTS: eta train_batch_size vali_batch_size seed")
      file.write("\nINPUTS: %f %d %d %d" %(eta,train_batch_size,vali_batch_size,seed))
      file.write("\nstep loss_train abserror_train loss_vali abserror_vali")
      file.close()

      print("Initializing training...")

      sess.run(tf.global_variables_initializer())
      time0 = time.time()

      for step in range(max_steps):
        x_batch, y_batch = create_batch(features,labels,train_batch_size)
        x_batch_vali, y_batch_vali = create_batch(features_vali,labels_vali,vali_batch_size)

        if step % log_frequency == 0:

            pred_error_train = pred_error.eval(feed_dict={x: x_batch, y: y_batch, keep_prob: 1})
            pred_error_vali = pred_error.eval(feed_dict={x: x_batch_vali, y: y_batch_vali, keep_prob: 1})
            loss_train = sess.run(loss,feed_dict={x: x_batch, y: y_batch, keep_prob: 1})
            loss_vali = sess.run(loss,feed_dict={x: x_batch_vali, y: y_batch_vali, keep_prob: 1})
            print('*** STEP %d ***' % step)
            print('Training - loss = %f, abserror = %.2f' % (loss_train,pred_error_train))
            print('Validation - loss = %f, abserror = %.2f' % (loss_vali,pred_error_vali))
            file = open("logs_loss/logs_%s_%s_%s_%s_%s.out" % (eta,n_fc,version,vali,max_steps),"a")
            file.write("\n%d %f %.2f %f %.2f" %(step,loss_train,pred_error_train,loss_vali,pred_error_vali))
            print("Losses successfully recorded!")

            print("%ss / %ssteps" % (time.time()-time0,log_frequency))
            time0 = time.time()

        train_step.run(feed_dict={x: x_batch_vali, y: y_batch_vali, keep_prob: 0.5})

        #stats = np.random.poisson(1.2, 3)
        #print("Level up! LVL %d: INT +%d, WIS +%d, DEX +%d, NEURONS +0" % (step, stats[0], stats[1], stats[2]))

      save_path = saver.save(sess, "nn_models/nn_model_%s_%s_%s_%s_%s.meta" % (eta,n_fc,version,vali,max_steps))
      print("Model saved in file: %s" % save_path)

  weights = []
  for ii in range(0,n_fc):
      with tf.variable_scope('fc_%d' %ii):
          # w_ = tf.get_variable("Variable:0",[1])
          ww = sess.run(eval('W_fc_%d' %ii))
          weights.append(ww)
  w_out = sess.run(W_out)
  bias_out = sess.run(b_out)

#############################################
# LOSS AND PREDICTION ERROR
#############################################
crs = open("logs_loss/logs_%s_%s_%s_%s_%s.out" % (eta,n_fc,version,vali,max_steps), "r")

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
plt.ylabel('Average prediction (absolute) error per batch')
plt.xlabel('Step')
plt.legend(loc=1)
plt.grid(True)
plt.show()


########################################
# ANALYZE WEIGHTS
########################################
with open('data/labels_NO_IMG.pickle','rb') as f:  # Python 3: open(..., 'rb')
    feats = pickle.load(f)

weights.append(w_out)
fig = plt.figure()
wide = 0.3*18/len(weights[0][0])
wd = (1-n_fc*wide)/(n_fc+1)
height = 0.8
xpos = 0.1
ypos = 0.1
wide = 0.3
height = 0.8
disp = 0.05
for ii in range(0,n_fc+1):
    ax = fig.add_subplot(1, n_fc+1, ii+1, position=[xpos,ypos,wide,height])
    xpos = xpos + wide + disp
    ypos = ypos + height/4
    wide = wide/2
    height = height/2
    im = ax.imshow(weights[ii],cmap='bwr')
    if ii==0:
        print(feats)
        ax.set_yticks(np.arange(len(feats)))
        ax.set_yticklabels(feats,fontsize=8)
    else:
        plt.axis('off')
    #plt.tight_layout()
#ax = fig.add_subplot(1, n_fc+1, n_fc+1)
#im = ax.imshow(w_out,cmap='bwr')
#plt.colorbar(im)
#plt.tight_layout()
plt.show()

# NORMALIZE WEIGHTSSSSS

if n_fc==0:
    plt.figure()#figsize=(10/2.54, 5/2.54), dpi=300)
    plt.stem(range(0,p),w_out)
    plt.xticks(range(0,p), feats, rotation='vertical')
    plt.rc('xtick',labelsize=8)
    plt.rc('ytick',labelsize=8)
    plt.tight_layout()
    plt.show()

########################################
# CLASSIFY TEST SAMPLES
########################################
def test_batch(features,labels,pos):
    # Shuffle data
    num_data = len(labels)
    order = np.linspace(0,num_data-1,num_data,dtype=int)
    features = features[order, ...]
    labels = labels[order]
    return features[pos:pos+1], labels[pos:pos+1]

preds_test = []
with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, "nn_models/nn_model_%s_%s_%s_%s_%s.meta" % (eta,n_fc,version,vali,max_steps))
    for i in range(0,len(features_test)):
        test_feat = features_test[i,...]
        test_label = labels_test[i]
        test_feat,test_label = test_batch(features_test,labels_test,i)
        prediction = sess.run(y_, feed_dict={x: test_feat, y: test_label, keep_prob: 1})
        preds_test.append(prediction)

labels_test = np.reshape(labels_test, len(features_test))
preds_test = np.reshape(preds_test, len(features_test))
n_show = 20
plt.scatter(range(0,len(labels_test[:n_show])),labels_test[:n_show],marker='s',label='Original value')
plt.scatter(range(0,len(labels_test[:n_show])),preds_test[:n_show],marker='x',label='Prediction')
plt.title('Some test examples')
plt.xlabel('Test sample')
plt.ylabel('Normalized dose')
plt.legend()
plt.show()

# plt.figure()
# test_pred_relerror = np.divide(np.abs(preds_test-labels_test),labels_test)
# plt.scatter(range(0,len(labels_test)),test_pred_relerror*100,marker='^')
# plt.xlabel('Test sample')
# plt.ylabel('Relative error in dose prediction (%)')
# plt.ylim([0,100])
# plt.show()

#fig, (ax1,ax2) = plt.subplots(1, 2, sharex='col', sharey='row')
fig = plt.figure(figsize=(6, 6))
grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
ax1 = fig.add_subplot(grid[:, :3])
ax2 = fig.add_subplot(grid[:, 3])

test_pred_abserror = np.abs(preds_test-labels_test)
ax1.scatter(range(0,len(labels_test)),test_pred_abserror,marker='^')
ax1.set_xlabel('Test sample')
ax1.set_ylabel('Absolute error in dose prediction')
#plt.ylim([0,1])
ax1.set_ylim([1e-6,1])
ax1.set_yscale('log')
#plt.show()

test_pred_abserror = np.abs(preds_test-labels_test)
ax2.boxplot(test_pred_abserror)
#plt.ylim([0,1])
ax2.set_ylim([1e-6,1])
ax2.set_yscale('log')
ax2.set_xticks([])
ax2.set_yticks([])
plt.show()
