from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.keras.backend as K
from keras_applications import correct_pad
from keras_applications import imagenet_utils
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, concatenate, BatchNormalization, Lambda, Input, multiply, add, ZeroPadding2D, Activation, Layer, MaxPooling2D, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras import Model
# import keras.backend as K
import tensorflow as tf
import numpy as np

RESIZE_FACTOR = 2

# TODO Change path to v1.1
BASE_WEIGHT_PATH = ('https://github.com/JonathanCMitchell/mobilenet_v2_keras/'
                    'releases/download/v1.1/')

backend = K

keras_utils = None

def resize_bilinear(x):
    return tf.image.resize_bilinear(x, size=[K.shape(x)[1]*RESIZE_FACTOR, K.shape(x)[2]*RESIZE_FACTOR])

def resize_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 4
    shape[1] *= RESIZE_FACTOR
    shape[2] *= RESIZE_FACTOR
    return tuple(shape)



def preprocess_input(x, **kwargs):
    """Preprocesses a numpy array encoding a batch of images.

    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].

    # Returns
        Preprocessed array.
    """
    return imagenet_utils.preprocess_input(x, mode='tf', **kwargs)

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def MobileNetV2(
                alpha=1.0,
                input_tensor=None,
                **kwargs):

    img_input = input_tensor if input_tensor is not None else layers.Input(shape=(None,None,3))
    first_block_filters = _make_divisible(32 * alpha, 8)
    x = layers.ZeroPadding2D(padding=correct_pad(backend, img_input, 3),
                             name='Conv1_pad')(img_input)
    x = layers.Conv2D(first_block_filters,
                      kernel_size=3,
                      strides=(2, 2),
                      padding='valid',
                      use_bias=False,
                      name='Conv1')(x)
    x = layers.BatchNormalization(
        epsilon=1e-3, momentum=0.999, name='bn_Conv1')(x)
    x = layers.ReLU(6., name='Conv1_relu')(x)

    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                            expansion=1, block_id=0)

    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                            expansion=6, block_id=1)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                            expansion=6, block_id=2)

    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                            expansion=6, block_id=3)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=4)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=5)

    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=2,
                            expansion=6, block_id=6)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=7)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=8)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=9)

    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=10)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=11)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=12)

    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=2,
                            expansion=6, block_id=13)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                            expansion=6, block_id=14)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                            expansion=6, block_id=15)

    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1,
                            expansion=6, block_id=16)

    # no alpha applied to last conv as stated in the paper:
    # if the width multiplier is greater than 1 we
    # increase the number of output channels
    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280

    x = layers.Conv2D(last_block_filters,
                      kernel_size=1,
                      use_bias=False,
                      name='Conv_1')(x)
    x = layers.BatchNormalization(epsilon=1e-3,
                                  momentum=0.999,
                                  name='Conv_1_bn')(x)
    x = layers.ReLU(6., name='out_relu')(x)


    return Model(input_tensor,x)


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
    in_channels = backend.int_shape(inputs)[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'block_{}_'.format(block_id)

    if block_id:
        # Expand
        x = layers.Conv2D(expansion * in_channels,
                          kernel_size=1,
                          padding='same',
                          use_bias=False,
                          activation=None,
                          name=prefix + 'expand')(x)
        x = layers.BatchNormalization(epsilon=1e-3,
                                      momentum=0.999,
                                      name=prefix + 'expand_BN')(x)
        x = layers.ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    # Depthwise
    if stride == 2:
        x = layers.ZeroPadding2D(padding=correct_pad(backend, x, 3),
                                 name=prefix + 'pad')(x)
    x = layers.DepthwiseConv2D(kernel_size=3,
                               strides=stride,
                               activation=None,
                               use_bias=False,
                               padding='same' if stride == 1 else 'valid',
                               name=prefix + 'depthwise')(x)
    x = layers.BatchNormalization(epsilon=1e-3,
                                  momentum=0.999,
                                  name=prefix + 'depthwise_BN')(x)

    x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = layers.Conv2D(pointwise_filters,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      activation=None,
                      name=prefix + 'project')(x)
    x = layers.BatchNormalization(
        epsilon=1e-3, momentum=0.999, name=prefix + 'project_BN')(x)

    if in_channels == pointwise_filters and stride == 1:
        return layers.Add(name=prefix + 'add')([inputs, x])
    return x

class EAST_model:
    def __init__(self, input_size=512):
        '''
        输入部分, 训练时有除了input_image以外, 还有
        overly_small_text_region_training_mask : 标记尺寸过小的文字, 避免干扰训练
        text_region_boundary_training_mask: 标记文本框与score map的边界
        target_score_map: grand truth score
        '''
        input_image = Input(shape=(None, None, 3), name='input_image')
        overly_small_text_region_training_mask = Input(shape=(None, None, 1), name='overly_small_text_region_training_mask')
        text_region_boundary_training_mask = Input(shape=(None, None, 1), name='text_region_boundary_training_mask')
        target_score_map = Input(shape=(None, None, 1), name='target_score_map')

        resnet = MobileNetV2(input_tensor=input_image)
        x = resnet.get_layer('out_relu').output

        x = Lambda(resize_bilinear, name='resize_1')(x)
        x = concatenate([x, resnet.get_layer('block_13_expand_relu').output], axis=3)
        x = Conv2D(128, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)
        x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)

        x = Lambda(resize_bilinear, name='resize_2')(x)
        x = concatenate([x, resnet.get_layer('block_6_expand_relu').output], axis=3)
        x = Conv2D(64, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)

        x = Lambda(resize_bilinear, name='resize_3')(x)
        x = concatenate([x, resnet.get_layer('block_3_expand_relu').output], axis=3)
        x = Conv2D(32, (1, 1), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)
        x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)

        x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-5))(x)
        x = BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
        x = Activation('relu')(x)

        pred_score_map = Conv2D(1, (1, 1), activation=tf.nn.sigmoid, name='pred_score_map')(x)
        rbox_geo_map = Conv2D(4, (1, 1), activation=tf.nn.sigmoid, name='rbox_geo_map')(x)
        rbox_geo_map = Lambda(lambda x: x * input_size)(rbox_geo_map)
        angle_map = Conv2D(1, (1, 1), activation=tf.nn.sigmoid, name='rbox_angle_map')(x)
        angle_map = Lambda(lambda x: (x - 0.5) * np.pi / 2)(angle_map)
        pred_geo_map = concatenate([rbox_geo_map, angle_map], axis=3, name='pred_geo_map')

        model = Model(inputs=[input_image, overly_small_text_region_training_mask, text_region_boundary_training_mask, target_score_map], outputs=[pred_score_map, pred_geo_map])

        self.model = model
        self.input_image = input_image
        self.overly_small_text_region_training_mask = overly_small_text_region_training_mask
        self.text_region_boundary_training_mask = text_region_boundary_training_mask
        self.target_score_map = target_score_map
        self.pred_score_map = pred_score_map
        self.pred_geo_map = pred_geo_map

if __name__=="__main__":
    east_mobile = EAST_model().model
    a = np.ones(shape = (8,1024,1024,3))
    b = np.ones(shape = (8,1024,1024,1))


