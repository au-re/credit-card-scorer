import tensorflow as tf
import numpy as np
import preprocess

'''
	ccscorer
	
	A single layer perceptron for assessing credit card scores using 
    tensorflow. https://www.tensorflow.org
	
	version: 0.1
	authors: Aurelien Hontabat
	license: MIT
'''

# our data (next time better to use tensorflows preprocessing tools)
data, results = preprocess.run()
input_dim = 20
output_dim = 2

# stay flexible when building the graph (tensorflow)
sess = tf.InteractiveSession()

# input layer
x = tf.placeholder(tf.float32, shape=[None, input_dim])

# output layer
y_ = tf.placeholder(tf.float32, shape=[None, output_dim])

# weights fo all connections
W = tf.Variable(tf.zeros([input_dim,output_dim]))

# bias
b = tf.Variable(tf.zeros([output_dim]))

# initialize the variables (tensorflow)
sess.run(tf.initialize_all_variables())

# our regression model: weights * input + bias
# additionally calculate the softmax probability for each class
y = tf.nn.softmax(tf.matmul(x,W) + b)

# our cost function 
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# training the model -----------------------------------------------------------

# defining the training function
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# run the training on some samples
for (d, r) in zip(data[:800], results[:800]):
	d = np.reshape(d, (-1, 20))
	r = np.reshape(r, (-1, 2))
	sess.run(train_step, feed_dict={x: d, y_: r})
	
print('training complete...')

# verifying results ------------------------------------------------------------

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# setup the test data
test_data = data[801:]
test_results = results[801:]

# print the accuracy on the test data
print(accuracy.eval(feed_dict={x: test_data, y_: test_results}))

# predicting the values from the test samples, note: the problem is unbalanced
prediction=tf.argmax(y,1)
print(prediction.eval(feed_dict={x: test_data}))

