from base import *
from config import *
import tensorflow as tf
import tflearn

class ResNeXt(object):
    def __init__(self, training):
        self.training = training
        self.x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, 3])

    def residual_block(self, inputs, planes, num_group, down_sample=False, stride=1, name='rb'):
        residual = inputs
        conv1 = conv_layer(inputs, planes*2, kernel=1, name=name+'_conv1')
        bn1 = batch_normalization(conv1, self.training, name=name+'_bn1')
        relu1 = relu(bn1)

        input_groups = tf.split(axis=3, num_or_size_splits=num_group, value=relu1)
        output_groups = [conv_layer(in_value, filter=planes*2, kernel=3, stride=stride, name=name+'_conv2_'+str(i))for i, in_value in enumerate(input_groups)]
        conv2 = concatenation(output_groups)
        bn2 = batch_normalization(conv2, self.training, name+'_bn2')
        relu2 = relu(bn2)

        conv3 = conv_layer(relu2, planes*4, kernel=1, name=name+'_conv3')
        bn3 = batch_normalization(conv3, self.training, name=name+'_bn3')

        if down_sample:
            residual = conv_layer(residual, planes*4, kernel=1, stride=stride, name=name+'_residual')
            residual = batch_normalization(residual, self.training)
        out = residual + bn3
        out = relu(out)
        return out

    def bottleneck(self, inputs, planes, blocks, num_group, stride, name=''):
        inputs = self.residual_block(inputs, planes, num_group, down_sample=True, stride=stride, name='{}_block_{}'.format(name, 1))
        for i in range(1, blocks):
            inputs = self.residual_block(inputs, planes, num_group, stride=1, name='{}_block_{}'.format(name, i+1))
        return inputs

    def make_layer(self, block, inputs, planes, blocks, num_group, stride=1, name=''):
        return block(inputs, planes, blocks, num_group, stride, name=name)

    def build(self, layers, num_group=32, num_classes=1000):
        conv1 = conv_layer(self.x, 64, [7, 7], 2, name='conv1')
        bn1 = batch_normalization(conv1, self.training, name='bn1')
        relu1 = relu(bn1)
        pool1 = max_pooling(relu1, 3, stride=2)

        layer1 = self.make_layer(self.bottleneck, pool1, 64, layers[0], num_group, name='l1')
        print('layer1:', layer1)
        layer2 = self.make_layer(self.bottleneck, layer1, 128, layers[1], num_group, stride=2, name='l2')
        print('layer2:', layer2)
        layer3 = self.make_layer(self.bottleneck, layer2, 256, layers[2], num_group, stride=2, name='l3')
        print('layer3:', layer3)
        layer4 = self.make_layer(self.bottleneck, layer3, 512, layers[3], num_group, stride=2, name='l4')
        print('layer4:', layer4)
        avg_pool = tf.layers.average_pooling2d(layer4,
                                               strides=[layer4.get_shape()[1], layer4.get_shape()[2]],
                                               pool_size=[layer4.get_shape()[1], layer4.get_shape()[2]],
                                               padding='SAME')
        print('avg_pool:', avg_pool)
        self.fc = tf.layers.dense(avg_pool, num_classes)
        print('fc', self.fc)

net = ResNeXt(True)
net.build([3, 4, 23, 3])










