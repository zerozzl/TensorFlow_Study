import tensorflow as tf

from tensorflow.contrib.session_bundle import exporter
from tensorflow.contrib.learn.python.learn.datasets import mnist as mnist_input_data

tf.app.flags.DEFINE_integer('training_iteration', 100,
                            'number of training iterations.')
tf.app.flags.DEFINE_integer('export_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', '../tutorials/MNIST_data/', 'Working directory.')
tf.app.flags.DEFINE_string('export_path', 'models/', 'Model directory.')
FLAGS = tf.app.flags.FLAGS


def main(_):
    # Train model
    print('Training model...')
    mnist = mnist_input_data.read_data_sets(FLAGS.work_dir, one_hot=True)
    x = tf.placeholder(tf.float32, [None, 784], name='x')
    y_ = tf.placeholder('float', shape=[None, 10])
    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    y = tf.nn.softmax(tf.matmul(x, w) + b, name='y')
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    for _ in range(FLAGS.training_iteration):
        batch = mnist.train.next_batch(50)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    print('training accuracy %g' % sess.run(accuracy,
                                            feed_dict={x: mnist.test.images,
                                                       y_: mnist.test.labels}))
    print('Done training!')

    # Export model
    export_path = FLAGS.export_path
    print('Exporting trained model to %s' % export_path)
    saver = tf.train.Saver(sharded=True)
    model_exporter = exporter.Exporter(saver)
    model_exporter.init(
        sess.graph.as_graph_def(),
        named_graph_signatures={
            'inputs': exporter.generic_signature({'images': x}),
            'outputs': exporter.generic_signature({'scores': y})})
    model_exporter.export(export_path, tf.constant(FLAGS.export_version), sess)
    print('Done exporting!')


if __name__ == '__main__':
    tf.app.run()
