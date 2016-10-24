import tensorflow as tf

''' Creation '''
"""
# Create two variables
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35), name="weights")
biases = tf.Variable(tf.zeros([200]), name="biases")
"""

''' Device placement '''
"""
# Pin a variable to CPU
with tf.device("/cpu:0"):
    v = tf.Variable(tf.zeros([1]))

# Pin a variable to GPU
with tf.device("/gpu:0"):
    v = tf.Variable(tf.zeros([1]))

# Pin a variable to a particular parameter server task
with tf.device("/job:ps/task:7"):
    v = tf.Variable(tf.zeros([1]))
"""

''' Initialization '''
"""
# Create two variables
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35), name="weights")
biases = tf.Variable(tf.zeros([200]), name="biases")

# Add an op to initialize the variables.
init_op = tf.initialize_all_variables()

# Later, when launching the model
with tf.Session() as sess:
    # Run the init operation.
    sess.run(init_op)
"""

''' Initialization from another Variable '''
"""
# Create a variable with a random value.
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35), name="weights")

# Create another variable with the same value as 'weights'.
w2 = tf.Variable(weights.initialized_value(), name="w2")

# Create another variable with twice the value of 'weights'
w_twice = tf.Variable(weights.initialized_value() * 2.0, name="w_twice")
"""

''' Saving and Restoring '''
"""
# Saving Variables
v1 = tf.Variable(tf.zeros([2]), name="v1")
v2 = tf.Variable(tf.ones([2]), name="v2")

init_op = tf.initialize_all_variables()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    save_path = saver.save(sess, "files/model.ckpt")
    print "Model saved in file: %s" % save_path
"""

"""
# Restoring Variables
v1 = tf.Variable(tf.zeros([2]), name="v1")
v2 = tf.Variable(tf.ones([2]), name="v2")

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "files/model.ckpt")
    print "Model restored."
"""

# Choosing which Variables to Save and Restore
v1 = tf.Variable(tf.zeros([2]), name="v1")
v2 = tf.Variable(tf.ones([2]), name="v2")

# Add ops to save and restore only 'v2' using the name "my_v2"
saver = tf.train.Saver({"my_v2:": v2})
