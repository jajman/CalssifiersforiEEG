import os, pickle, random
from spectrum import Periodogram
import tensorflow as tf
import numpy as np
import time
sEEG=[]
nsEEG=[]
spect_sEEG=[]
spect_sAns=[]
spect_nsEEG=[]
spect_nsAns=[]
save_path='./FP'
dirName='AUTOANNOT2/'
train_r=0.9
val_r=0.1

batch_size=256
load_model=False

with open(dirName+'sEEGData.bin', 'rb') as fp:
    sEEG = pickle.load(fp)
s_full_len = len(sEEG)

with open(dirName+'nsEEGData.bin', 'rb') as fp:
    nsEEG = pickle.load(fp)
ns_full_len = len(nsEEG)

num_units=200

with tf.name_scope("seizure"):
    x = tf.placeholder(tf.float32, [None, 100])

    w2 = tf.Variable(tf.truncated_normal([100, num_units]))
    b2 = tf.Variable(tf.zeros([num_units]))
    hidden2 = tf.nn.relu(tf.matmul(x, w2) + b2)

    w1 = tf.Variable(tf.truncated_normal([num_units, num_units//4]))
    b1 = tf.Variable(tf.zeros([ num_units//4]))
    hidden1 = tf.nn.relu(tf.matmul(hidden2, w1) + b1)

    w0 = tf.Variable(tf.zeros([num_units//4, 2]))
    b0 = tf.Variable(tf.zeros([2]))

    pp = tf.nn.softmax(tf.matmul(hidden1, w0) + b0)
    pp=pp+0.000000001
    t = tf.placeholder(tf.float32, [None, 2])
    loss = -tf.reduce_sum(t * tf.log(pp))
    train_step = tf.train.AdamOptimizer().minimize(loss)

    correct_prediction = tf.equal(tf.argmax(pp, 1), tf.argmax(t, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
saver=tf.train.Saver(max_to_keep=10)

if not os.path.exists(save_path):
    os.makedirs(save_path)

for i in range(s_full_len):
    EEGtoSpect = sEEG[i]
    p = Periodogram(EEGtoSpect, sampling=1000)
    spect_sEEG.append(p.psd[0:100])
    spect_sAns.append([1,0])

for i in range(ns_full_len):
    EEGtoSpect = nsEEG[i]
    p = Periodogram(EEGtoSpect, sampling=1000)
    spect_nsEEG.append(p.psd[0:100])
    spect_nsAns.append([0,1])


random.shuffle(spect_sEEG)
random.shuffle(spect_nsEEG)

s_train_num=int(len(spect_sEEG)*train_r)
s_val_num=int(len(spect_sEEG)*val_r)
ns_train_num=int(len(spect_nsEEG)*train_r)
ns_val_num=int(len(spect_nsEEG)*val_r)

ns_val = spect_nsEEG[0:ns_val_num]
ns_train=spect_nsEEG[ns_val_num:]
ns_ans_val = spect_nsAns[0:ns_val_num]
ns_ans_train = spect_nsAns[ns_val_num:]

s_train = spect_sEEG[0:s_train_num]
s_val = spect_sEEG[s_train_num:]
s_ans_train = spect_sAns[0:s_train_num]
s_ans_val = spect_sAns[s_train_num:]

s_ans_train.extend(ns_ans_train)
s_ans_val.extend(ns_ans_val)
s_train.extend(ns_train)
s_val.extend(ns_val)

s_ans_train = np.array(s_ans_train)
s_ans_val = np.array(s_ans_val)
s_train = np.array(s_train)
s_val = np.array(s_val)


index=[]
for i in range(len(s_train)):
    index.append(i)
turn=len(s_train)//batch_size

if load_model:
    print('loading model...')
    ckpt=tf.train.get_checkpoint_state(save_path)
    path=ckpt.model_checkpoint_path
    saver.restore(sess,ckpt.model_checkpoint_path)
loss_val, acc_val = sess.run([loss, accuracy],
                                 feed_dict={x: s_val, t: s_ans_val})
print('pretrain , Loss: %f, Accuracy: %f'
              % ( loss_val, acc_val))
time.sleep(2)
startTime = time.time()
for i in range(100000):
    random.shuffle(index)

    for j in range(turn):
        train_batch=s_train[index[j * batch_size:(j + 1) * batch_size]]
        ans_batch=s_ans_train[index[j * batch_size:(j + 1) * batch_size]]
        sess.run(train_step, feed_dict={x: train_batch, t: ans_batch})
    if i % 10 == 0:

        loss_val, acc_val = sess.run([loss, accuracy],
                                     feed_dict={x: s_val, t: s_ans_val})
        loss_train, acc_train = sess.run([loss, accuracy],
                                     feed_dict={x: s_train, t: s_ans_train})
        print('Step: %d, Train Loss: %f, Train Accuracy: %f, Val Loss: %f, Val Accuracy: %f'
              % (i, loss_train, acc_train, loss_val, acc_val))
        pred = sess.run(pp, feed_dict={x: s_val, t: s_ans_val})
        '''for k in range(15):
            print(pred[k])'''
    if i%50==0 and i!=0:

        saver.save(sess, save_path + '/model-' + str(i) + '.cptk')
        print('reshuffle')

    if i >= 1000:
        break

endTime = time.time()
print(str(endTime - startTime) + ' SEC')
