import os, pickle, random, sys

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
save_name='./cnn2-40'+str(argVal)+'.txt'
save_path='./cnn2-40'+str(argVal)

dirName='AUTOANNOT3/'
train_r=0.9
val_r=0.1

batch_size=128
load_model=False

WHITE = (255, 255, 255)

with open('CNNsEEG40.bin', 'rb') as fp:
    spect_sEEG=pickle.load(fp)
s_full_len=len(spect_sEEG)
for k in range(s_full_len):
    spect_sAns.append([1, 0])
numEEG=3
for i in range(numEEG):
    fileName='CNNnsEEG40-'+str(i)+'.bin'
    with open(fileName, 'rb') as fp:
        spect_nsEEG+=pickle.load(fp)
ns_full_len=len(spect_nsEEG)
for k in range(ns_full_len):
    spect_nsAns.append([0,1])



with tf.name_scope("CNN"):
    num_filters1 = 32

    cx = tf.placeholder(tf.float32, [None, 40,250])
    x_image = tf.reshape(cx, [-1, 40, 250, 1])

    W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, num_filters1],
                                              stddev=0.1))
    h_conv1 = tf.nn.conv2d(x_image, W_conv1,
                           strides=[1, 1, 1, 1], padding='SAME')

    b_conv1 = tf.Variable(tf.constant(0.1, shape=[num_filters1]))
    h_conv1_cutoff = tf.nn.relu(h_conv1 + b_conv1)

    h_pool1 = tf.nn.max_pool(h_conv1_cutoff, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], padding='SAME')

    num_filters2 = 64

    W_conv2 = tf.Variable(
        tf.truncated_normal([5, 5, num_filters1, num_filters2],
                            stddev=0.1))
    h_conv2 = tf.nn.conv2d(h_pool1, W_conv2,
                           strides=[1, 1, 1, 1], padding='SAME')

    b_conv2 = tf.Variable(tf.constant(0.1, shape=[num_filters2]))
    h_conv2_cutoff = tf.nn.relu(h_conv2 + b_conv2)

    h_pool2 = tf.nn.max_pool(h_conv2_cutoff, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], padding='SAME')
    h_pool2_flat = tf.reshape(h_pool2, [-1, 10 * 63 * num_filters2])

    num_units1 = 10 * 63 * num_filters2
    num_units2 = 512
    num_units3 = 128
    final_output_size = 2
    keep_prob = 0.7
    train_mode = tf.placeholder(tf.bool, name='train_mode')
    xavier_init = tf.contrib.layers.xavier_initializer()
    bn_params = {
        'is_training': train_mode,
        'decay': 0.9,
        'updates_collections': None
    }

    # We can build short code using 'arg_scope' to avoid duplicate code
    # same function with different arguments
    with arg_scope([fully_connected],
                   activation_fn=tf.nn.relu,
                   weights_initializer=xavier_init,
                   biases_initializer=None,
                   normalizer_fn=batch_norm,
                   normalizer_params=bn_params
                   ):
        hidden_layer1 = fully_connected(h_pool2_flat, num_units2, scope="h1")
        h1_drop = dropout(hidden_layer1, keep_prob, is_training=train_mode)
        hidden_layer2 = fully_connected(h1_drop, num_units3, scope="h2")
        h2_drop = dropout(hidden_layer2, keep_prob, is_training=train_mode)
        hypothesis = fully_connected(h2_drop, final_output_size, activation_fn=None, scope="hypothesis")

    cp = tf.nn.softmax(hypothesis)
    cp = cp + 0.000000001
    ct = tf.placeholder(tf.float32, [None, 2])
    closs = -tf.reduce_sum(ct * tf.log(cp))
    ctrain_step = tf.train.AdamOptimizer(0.0001).minimize(closs)
    ccorrect_prediction = tf.equal(tf.argmax(cp, 1), tf.argmax(ct, 1))
    caccuracy = tf.reduce_mean(tf.cast(ccorrect_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
saver=tf.train.Saver(max_to_keep=10)

if not os.path.exists(save_path):
    os.makedirs(save_path)



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
s_val = np.array(s_val)

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

if load_model:
    print('loading model...')
    ckpt=tf.train.get_checkpoint_state(save_path)
    path=ckpt.model_checkpoint_path
    saver.restore(sess,ckpt.model_checkpoint_path)
loss_val, acc_val = sess.run([closs, caccuracy],
                                 feed_dict={cx: s_val[:1000], ct: s_ans_val[:1000], train_mode: False})
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
        sess.run(ctrain_step, feed_dict={cx: train_batch, ct: ans_batch, train_mode: True})

    if i % 5 == 0:

        loss_val, acc_val = sess.run([closs, caccuracy],
                                     feed_dict={cx: s_val[:1000], ct: s_ans_val[:1000], train_mode: False})
        if acc_val>maxAcc and i>5:
            maxAcc=acc_val
            minLoss = loss_val
            saver.save(sess, save_path + '/model-' + str(i) + '.cptk')
        if acc_val==maxAcc and i>10:
            if minLoss>loss_val:
                minLoss=loss_val
                saver.save(sess, save_path + '/model-' + str(i) + '.cptk')


        loss_train, acc_train = sess.run([closs, caccuracy],
                                         feed_dict={cx: s_train[:1000], ct: s_ans_train[:1000], train_mode: False})
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
