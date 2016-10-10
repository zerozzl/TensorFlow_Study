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

# Variables
state = tf.Variable(0, name="counter")
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_op)
    print sess.run(state)

    for _ in range(3):
        sess.run(update)
        print sess.run(state)

# Fetches
input1 = tf.constant([3.])
input2 = tf.constant([2.])
input3 = tf.constant([5.])
intermed = tf.add(input2, input3)
mul = tf.mul(input1, intermed)

with tf.Session() as sess:
    result = sess.run([mul, intermed])
    print result

# Feeds
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1, input2)

with tf.Session() as sess:
    print sess.run([output], feed_dict={input1: [7.], input2: [2.]})
