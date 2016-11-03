import tensorflow as tf
import cifar10_input


class CIFAR10InputTest(tf.test.TestCase):

    def _record(self, label, red, green, blue):
        image_size = 32 * 32
        record = bytes(bytearray([label] + [red] * image_size +
                                 [green] * image_size + [blue] * image_size))
        expected = [[[red, green, blue]] * 32] * 32
        return record, expected
