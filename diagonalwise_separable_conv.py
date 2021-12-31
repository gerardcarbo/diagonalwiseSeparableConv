import tensorflow as tf
from tensorflow.keras.layers import (
    Layer,
)
import numpy as np
from absl import app
import traceback

@tf.keras.utils.register_keras_serializable()
class DiagonalwiseSeparableLayer(Layer):

    def __init__(self, kernel_size, out_channels, stride, padding, group_size, **kwargs):
        super(DiagonalwiseSeparableLayer, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.group_size = group_size

    def get_mask(self, in_channels, kernel_size):
        mask = np.zeros((kernel_size, kernel_size, in_channels, in_channels))
        for _ in range(in_channels):
            mask[:, :, _, _] = 1.
        return tf.constant(mask, dtype='float32')

    def build(self, input_shape):
        #diagonalwise
        self.in_channels = input_shape[-1]
        self.groups = int(max(self.in_channels / self.group_size, 1))
        channels = int(self.in_channels / self.groups)

        self.mask = self.get_mask(channels, self.kernel_size)

        self.splitw = [self.add_weight(name = "diagwConv"+str(i), shape=(self.kernel_size, self.kernel_size, channels, channels), trainable=True) for i in range(self.groups)]

        #pointwise
        self.pw = self.add_weight(name = "pointwConv", shape = (1, 1, channels, self.out_channels), trainable=True)

    @tf.function
    def call(self, x):
        #diagonalwise
        splitx = tf.split(x, self.groups, -1)
        splitx = [tf.nn.conv2d(x, tf.multiply(w, self.mask), (1, self.stride, self.stride, 1), self.padding)
                  for x, w in zip(splitx, self.splitw)]
        x = tf.concat(splitx, -1)

        # pointwise
        x = tf.nn.conv2d(x, self.pw, (1, 1, 1, 1), self.padding)
        return x

    def get_config(self):
        config = super(DiagonalwiseSeparableLayer, self).get_config()
        config.update(
            {'kernel_size': self.kernel_size,
            'out_channels': self.out_channels,
            'group_size': self.group_size,
            'stride': self.stride,
            'padding': self.padding})
        return config

def main(_argv):

    conv = DiagonalwiseSeparableLayer(3,32,1,'SAME',True)

    x = conv(tf.ones((16,64,64,16)))
    tf.print(x.shape)

if __name__ == '__main__':
    try:
        app.run(main)
    except Exception as e:
        print('train EXCEPTION', e)
        traceback.print_exc()
        pass
