import os, pickle, random, math, sys
from spectrum import Periodogram
import tensorflow as tf
import numpy as np
import time
from tensorflow.contrib.layers import fully_connected, batch_norm, dropout
from tensorflow.contrib.framework import arg_scope
sEEG=[]
nsEEG=[]
spect_sEEG=[]
spect_sAns=[]
spect_nsEEG=[]
spect_nsAns=[]
try:
    if sys.argv[1]:
        argVal = sys.argv[1]
except:
    argVal = 0
save_name='./rnnSpectro'+str(argVal)+'.txt'
save_path='./rnnSpectro'+str(argVal)

dirName='AUTOANNOT5BU/'
train_r=0.9
val_r=0.1

batch_size=256
load_model=False


n_in=2
n_out=2
n_hidden=20

with open(dirName+'sEEGData.bin', 'rb') as fp:
    sEEG = pickle.load(fp)
s_full_len = len(sEEG)

with open(dirName+'nsEEGData.bin', 'rb') as fp:
    nsEEG = pickle.load(fp)
ns_full_len = len(nsEEG)

for i in range(s_full_len):
    EEGtoSpect = sEEG[i]
    p = Periodogram(EEGtoSpect, sampling=1000)
    spect_sEEG.append(p.psd[0:100])
    spect_sAns.append([1,0])

for i in range(ns_full_len):
    EEGtoSpect = nsEEG[i]
    p = Periodogram(EEGtoSpect, sampling=1000)
    spect_nsEEG.append(p.psd[0:100])
    spect_nsAns.append([0, 1])

random.shuffle(spect_sEEG)
random.shuffle(spect_nsEEG)

s_train_num=int(len(spect_sEEG)*train_r)
s_val_num=len(spect_sEEG)-s_train_num
ns_train_num=int(len(spect_nsEEG)*train_r)
ns_val_num=len(spect_nsEEG)-ns_train_num

s_train = spect_sEEG[0:s_train_num]
ns_train = spect_nsEEG[0:ns_train_num]

s_val = spect_sEEG[s_train_num:]
ns_val = spect_nsEEG[ns_train_num:]

s_ans_train = spect_sAns[0:s_train_num]
ns_ans_train = spect_nsAns[0:ns_train_num]

s_ans_val = spect_sAns[s_train_num:]
ns_ans_val = spect_nsAns[ns_train_num:]

s_ans_train.extend(ns_ans_train)
s_ans_val.extend(ns_ans_val)
s_train.extend(ns_train)
s_val.extend(ns_val)

s_ans_train = np.array(s_ans_train)
s_ans_val = np.array(s_ans_val)
s_train = np.array(s_train)
s_train_len=s_train.shape[0]
s_train=s_train.reshape([s_train_len,-1,n_in])
s_val = np.array(s_val)
s_val_len=s_val.shape[0]
s_val=s_val.reshape([s_val_len,-1,n_in])

maxlen = s_train.shape[1]
print('maxlen='+str(maxlen))

index=[]
for i in range(len(s_val)):
    index.append(i)
random.shuffle(index)
s_val=s_val[index]
s_ans_val=s_ans_val[index]


index=[]
for i in range(len(s_train)):
    index.append(i)
turn=len(s_train)//batch_size

def inference(x, n_batch, maxlen=None, n_hidden=None, n_out=None):
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.zeros(shape, dtype=tf.float32)
        return tf.Variable(initial)

    cell = tf.contrib.rnn.LSTMCell(n_hidden, forget_bias=1.0)
    initial_state = cell.zero_state(n_batch, tf.float32)

    state = initial_state
    outputs = []  
    with tf.variable_scope('LSTM'):
        for t in range(maxlen):
            if t > 0:
                tf.get_variable_scope().reuse_variables()
            (cell_output, state) = cell(x[:, t, :], state)
            outputs.append(cell_output)

    output = outputs[-1]

    V = weight_variable([n_hidden, n_out])
    c = bias_variable([n_out])
    y = tf.matmul(output, V) + c  

    return y


def loss(y, t):
    cp = tf.nn.softmax(y)
    cp = cp + 0.000000001
    closs = -tf.reduce_sum(t * tf.log(cp))
    return closs


def training(loss):
    optimizer = \
        tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)

    train_step = optimizer.minimize(loss)
    return train_step



with tf.name_scope("RNN500"):
    cx = tf.placeholder(tf.float32, shape=[None, maxlen, n_in])
    ct = tf.placeholder(tf.float32, shape=[None, n_out])
    n_batch = tf.placeholder(tf.int32, shape=[])


    y = inference(cx, n_batch, maxlen=maxlen, n_hidden=n_hidden, n_out=n_out)
    closs = loss(y, ct)
    ctrain_step = training(closs)

    ccorrect_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(ct, 1))
    caccuracy = tf.reduce_mean(tf.cast(ccorrect_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
saver=tf.train.Saver(max_to_keep=10)

if not os.path.exists(save_path):
    os.makedirs(save_path)


if load_model:
    print('loading model...')
    ckpt=tf.train.get_checkpoint_state(save_path)
    path=ckpt.model_checkpoint_path
    saver.restore(sess,ckpt.model_checkpoint_path)
loss_val, acc_val = sess.run([closs, caccuracy],
                                 feed_dict={cx: s_val[:256], ct: s_ans_val[:256], n_batch:batch_size})
print('pretrain , Loss: %f, Accuracy: %f'
              % ( loss_val, acc_val))

time.sleep(2)
startTime = time.time()
maxAcc=0
minLoss=10000
logStr=''
for i in range(10000):
    random.shuffle(index)
    for j in range(turn):

        train_batch=s_train[index[j * batch_size:(j + 1) * batch_size]]
        ans_batch=s_ans_train[index[j * batch_size:(j + 1) * batch_size]]
        sess.run(ctrain_step, feed_dict={cx: train_batch, ct: ans_batch, n_batch:batch_size})

    if i % 5 == 0:

        loss_val, acc_val = sess.run([closs, caccuracy],
                                     feed_dict={cx: s_val[:3000], ct: s_ans_val[:3000],n_batch:2033})
        if acc_val>maxAcc and i>5:
            maxAcc=acc_val
            minLoss = loss_val
            saver.save(sess, save_path + '/model-' + str(i) + '.cptk')
        if acc_val==maxAcc and i>10:
            if minLoss>loss_val:
                minLoss=loss_val
                saver.save(sess, save_path + '/model-' + str(i) + '.cptk')
        loss_train, acc_train = sess.run([closs, caccuracy],
                                         feed_dict={cx: s_train[:1000], ct: s_ans_train[:1000], n_batch:1000})
        print('Step: %d, Train Loss: %f, Train Accuracy: %f, Val Loss: %f, Val Accuracy: %f'
              % (i, loss_train, acc_train, loss_val, acc_val))
        logStr += 'Step: %d, Train Loss: %f, Train Accuracy: %f, Val Loss: %f, Val Accuracy: %f' % (
            i, loss_train, acc_train, loss_val, acc_val)
        logStr += '\n'

    if i >= 1000:
        break

endTime = time.time()
print(str(endTime - startTime) + ' SEC')

with open(save_name, 'w') as f:
    f.write(logStr)
