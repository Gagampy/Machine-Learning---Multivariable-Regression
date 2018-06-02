import scipy.io as sio
from scipy import signal as sig
import numpy as np
import tensorflow as tf

matpath = 'C:/Users/Vlad/Desktop/Machine Learning/Projects/SystolicPressure/'
filename_sp = 'systolicpres.mat'
filename_params = 'ppg_params.mat'

systolic_pressure = sio.loadmat(matpath + filename_sp)
ppg_params = sio.loadmat(matpath + filename_params)

def feature_normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std

y_data = np.transpose(systolic_pressure.get('BP_l'))
y_train = y_data[0:500, :]
y_test = y_data[500:, :]

x_data = np.transpose(ppg_params.get('PeakSys_l'))
x_data = feature_normalize(x_data)
x_train = x_data[0:500, :]
x_test = x_data[500:, :]

print(x_data)

def hiddenlayer(x_input, input_size, output_size):
    W = tf.Variable(tf.truncated_normal([input_size, output_size], stddev = 0.01))
    b = tf.Variable(tf.truncated_normal([1], stddev = 0.01))
    return tf.matmul(x_input, W) + b

with tf.variable_scope('Initialization'):
    n_epoch, n_hidden = 1000000, 10
    n_row, n_col = np.shape(x_train)
    n_row_test, n_col_test = np.shape(x_test)
    n_input = n_row
    y_input = tf.placeholder(tf.float32, [None, 1])
    x_input = tf.placeholder(tf.float32, [None, n_col])
    h1 = tf.sigmoid(hiddenlayer(x_input, n_col, n_hidden))
    y_pred = hiddenlayer(h1, n_hidden, 1)


loss_function = tf.reduce_sum(tf.square(y_pred - y_input) / (2 * n_input))
train = tf.train.AdamOptimizer(0.008).minimize(loss_function)

loss_history = []
result = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epoch):
        _, loss = sess.run([train, loss_function], feed_dict={x_input: x_train, y_input: y_train})
        loss_history.append(loss)
        print("Epoch: ", epoch, ", loss: ", loss)

        if loss <= 10:
            break
    print(n_row_test, n_col_test)
    x_test = x_test.reshape(n_row_test, n_col_test)
    print(np.shape(x_test))
    for i in range(n_row_test):
        x_to_feed = x_test[i,:].reshape(1, n_col_test)
        result.append(sess.run(y_pred, feed_dict={x_input: x_to_feed}))


np.save('C:/Users/Vlad/Desktop/Machine Learning/Projects/SystolicPressure/' + 'loss_history.npy', loss_history)
np.save('C:/Users/Vlad/Desktop/Machine Learning/Projects/SystolicPressure/' + 'result_prediction.npy', result)