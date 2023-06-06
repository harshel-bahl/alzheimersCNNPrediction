from typing import Union

import tensorflow as tf

class ConvBlock(tf.keras.layers.Layer):
    """
    Convolutional layer, with batch normalization and an activation function.
    """
    def __init__(self, filters: int, kernel_size: tuple[int, int],
                 use_batch_norm: bool = True,
                 pool_size: tuple[int, int] = None,
                 activation: Union[tf.keras.layers.Activation, str] = "gelu",
                 name: str = "", **kwargs):
        super(ConvBlock, self).__init__(name=name, **kwargs)
        
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size, padding="same", name=name+"_conv")
        self.use_batch_norm = use_batch_norm
        self.pool_size = pool_size
        
        if self.use_batch_norm:
            self.bn = tf.keras.layers.BatchNormalization(name=name+"_bn")
            
        try:
            self.activation = activation(name=name + "_act")
        except TypeError:
            self.activation = tf.keras.layers.Activation(activation, name=name + "_act")
            
        if self.pool_size is not None:
            self.pooling = tf.keras.layers.MaxPool2D(pool_size=pool_size, name=name + "_pool")

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        out = self.conv(inputs)
        if self.use_batch_norm:
            out = self.bn(out)
            
        out = self.activation(out)
        if self.pool_size is not None:
            out = self.pooling(out)
        
        return out
    

class InceptionBlock(tf.keras.layers.Layer):
    """
    Inception module.
    """
    def __init__(self,
                 filters_1x1: int,
                 filters_3x3: int,
                 filters_5x5: int,
                 activation: Union[tf.keras.layers.Activation, str] = "gelu",
                 name: str = "", **kwargs):
        super(InceptionBlock, self).__init__(name=name, **kwargs)

        self.conv1x1 = ConvBlock(filters=filters_1x1, kernel_size=(1, 1),
                                 use_batch_norm=True, activation=activation, name=f"{name}_conv1x1")

        self.conv3x3_1 = ConvBlock(filters=filters_1x1, kernel_size=(1, 1),
                                   use_batch_norm=True, activation=activation, name=f"{name}_conv3x3_1")
        self.conv3x3_2 = ConvBlock(filters=filters_3x3, kernel_size=(3, 3),
                                   use_batch_norm=True, activation=activation, name=f"{name}_conv3x3_2")

        self.conv5x5_1 = ConvBlock(filters=filters_1x1, kernel_size=(1, 1),
                                   use_batch_norm=True, activation=activation, name=f"{name}_conv5x5_1")
        self.conv5x5_2 = ConvBlock(filters=filters_5x5, kernel_size=(5, 5),
                                   use_batch_norm=True, activation=activation, name=f"{name}_conv5x5_2")

        self.pooling_1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), padding="same", strides=(1, 1), name=f"{name}_pooling_1")
        self.pooling_2 = ConvBlock(filters=filters_1x1, kernel_size=(1, 1),
                                   use_batch_norm=True, activation=activation, name=f"{name}_pooling_2")

        self.concat = tf.keras.layers.Concatenate(axis=-1, name=f"{name}_concat")

    def call(self, inputs, **kwargs):
        conv_1x1_output = self.conv1x1(inputs)

        conv_3x3_output = self.conv3x3_1(inputs)
        conv_3x3_output = self.conv3x3_2(conv_3x3_output)

        conv_5x5_output = self.conv5x5_1(inputs)
        conv_5x5_output = self.conv5x5_2(conv_5x5_output)

        pooling_output = self.pooling_1(inputs)
        pooling_output = self.pooling_2(pooling_output)

        out = self.concat([conv_1x1_output, conv_3x3_output, conv_5x5_output, pooling_output])

        return out


class InceptionConvNet(tf.keras.Model):
    """
    A simple convolutional network classifier.
    """
    def __init__(self, **kwargs):
        super(InceptionConvNet, self).__init__(**kwargs)
        
        self.l1 = InceptionBlock(32, 32, 32, name="l1")
        self.l2 = InceptionBlock(16, 16, 16, name="l2")
        self.l3 = InceptionBlock(8, 8, 8, name="l3")
        self.l4 = InceptionBlock(1, 1, 1, name="l4")

        self.flatten = tf.keras.layers.Flatten()
        
        self.fc1 = tf.keras.layers.Dense(64, name="fc1")
        self.fc1_dropout = tf.keras.layers.Dropout(0.5)
        self.fc1_activation = tf.keras.layers.LeakyReLU()
        
        self.fc2 = tf.keras.layers.Dense(4, name="fc2")
        self.fc2_activation = tf.keras.activations.softmax
        
    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        out = self.l1(inputs)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
                
        out = self.flatten(out)
                
        out = self.fc1(out)
        out = self.fc1_dropout(out)
        out = self.fc1_activation(out)
        
        out = self.fc2(out)
        out = self.fc2_activation(out)
        return out