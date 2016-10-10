import tensorflow as tf

# Building the graph
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])
product = tf.matmul(matrix1, matrix2)

# Launching the graph in a session
sess = tf.Session()
result = sess.run(product)
print result
sess.close()

with tf.Session() as sess:
    result = sess.run([product])
    print result


with tf.Session() as sess:
    with tf.device("/cpu:0"):
        matrix1 = tf.constant([[3., 3.]])
        matrix2 = tf.constant([[2.], [2.]])
        product = tf.matmul(matrix1, matrix2)
        result = sess.run(product)
        print result


# Launching the graph in a distributed session
'''
with tf.Session("grpc://example.org:2222") as sess:
    result = sess.run(product)
'''

'''
with tf.device("/job:ps/task:0"):
    weights = tf.Variable("...")
    biases = tf.Variable("...")
'''

# Interactive Usage
'''
isess = tf.InteractiveSession()
x = tf.Variable([1., 2.])
a = tf.Variable([3., 3.])
x.initializer.run()

sub = tf.sub(x, a)
print sub.eval()
isess.close()
'''