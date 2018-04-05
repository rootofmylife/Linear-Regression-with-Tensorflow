import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50

# Y = aX + b => X, Y?
x_datas = [147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]
xdata_mean = np.mean(x_datas)
xdata_std = np.std(x_datas)

y_datas = [49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]
ydata_mean = np.mean(y_datas)
ydata_std = np.std(y_datas)

x_datas = [ (s - xdata_mean) / xdata_std for s in x_datas]
y_datas = [ (s - ydata_mean) / ydata_std for s in y_datas]

x_vals = np.array(x_datas)
y_vals = np.array(y_datas)

x_data = tf.placeholder(dtype= tf.float32)
y_target = tf.placeholder(dtype= tf.float32)

A = tf.Variable(np.random.randn())
b = tf.Variable(np.random.randn())

model_output = tf.add(tf.multiply(x_data, A), b)

# Loss function: Mean squared error
cost = tf.reduce_sum(tf.pow(model_output - y_target, 2)) / (2 * 13) # 13 là chiều dài của mảng data
# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    for epoch in range(training_epochs):
        for (x, y) in zip(x_vals, y_vals):
            sess.run(optimizer, feed_dict={x_data: x, y_target: y})
    
        if (epoch + 1) % display_step == 0:
            c = sess.run(cost, feed_dict={x_data: x_vals, y_target: y_vals})
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c), "W=", sess.run(A), "b=", sess.run(b))

    training_cost = sess.run(cost, feed_dict={x_data: x_vals, y_target: y_vals})
    print("Training cost=", training_cost, "W=", sess.run(A), "b=", sess.run(b), '\n')

    plt.plot(x_vals, y_vals, 'ro', label='Original data')
    plt.plot(x_vals, sess.run(A) * x_vals + sess.run(b), label='Fitted line')
    plt.xlabel('Height (cm)')
    plt.ylabel('Weight (kg)')
    plt.show()