import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

fname = ['rawdata_t.csv','lbldata_t.csv']
xy = np.loadtxt(fname[0],delimiter=',',dtype=np.float32)
lb = np.loadtxt(fname[1],delimiter=',',dtype=np.int32)

input_data = xy[:,:]
label_data = lb[:,:]

fname = ['rawdata_s.csv','lbldata_s.csv']
xy = np.loadtxt(fname[0],delimiter=',',dtype=np.float32)
lb = np.loadtxt(fname[1],delimiter=',',dtype=np.int32)

input_data_s = xy[:,:]
label_data_s = lb[:,:]

INPUT_SIZE = 100
HIDDEN1_SIZE = 100
HIDDEN2_SIZE = 100
CLASSES = 2
Learning_Rate = 0.001

x= tf.placeholder(tf.float32, shape = [None,INPUT_SIZE])
y_= tf.placeholder(tf.float32, shape = [None,CLASSES])
keep_prob = tf.placeholder(tf.float32)

tensor_map = {x: input_data, y_: label_data ,keep_prob: 0.7 }

w_h1 = tf.get_variable("w1",shape=[INPUT_SIZE,HIDDEN1_SIZE],initializer=tf.contrib.layers.xavier_initializer())
b_h1 = tf.Variable(tf.random_normal([HIDDEN1_SIZE]))
hidden1 = tf.nn.relu(tf.matmul(x,w_h1) + b_h1)
hidden1 = tf.nn.dropout(hidden1, keep_prob=keep_prob)

w_h2 = tf.get_variable("w2",shape=[HIDDEN1_SIZE,HIDDEN2_SIZE],initializer=tf.contrib.layers.xavier_initializer())
b_h2 = tf.Variable(tf.random_normal([HIDDEN2_SIZE]))
hidden2 = tf.nn.relu(tf.matmul(hidden1,w_h2) + b_h2)
hidden2 = tf.nn.dropout(hidden2, keep_prob=keep_prob)

w_o = tf.get_variable("o",shape=[HIDDEN2_SIZE,CLASSES],initializer=tf.contrib.layers.xavier_initializer())
b_o = tf.Variable(tf.random_normal([CLASSES]))
y = tf.matmul(hidden2,w_o) + b_o

# cost = tf.reduce_mean(-y_*tf.log(y) - (1-y_)*tf.log(1-y))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=y, labels=y_))
train = tf.train.AdamOptimizer(Learning_Rate).minimize(cost)

prediction = tf.arg_max(y,1)
is_correct = tf.equal(prediction, tf.arg_max(y_,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

# graph
accuracy_summ = tf.summary.scalar("accuracy", accuracy)
cost_summ = tf.summary.scalar("cost", cost)
summary = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs/test_logs_1')
writer.add_graph(sess.graph)


for i in range(2000):
    s, _, loss = sess.run([summary, train, cost],feed_dict = tensor_map)
    writer.add_summary(s, global_step=i)
    if i %100 == 0:
        print ("Step: ", i)
        print ("loss: ", loss)
        print("Accuracy: ", sess.run(accuracy, feed_dict={x:input_data_s, y_:label_data_s ,keep_prob: 1}))