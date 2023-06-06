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


class ConvNet(tf.keras.Model):
    """
    A simple convolutional network classifier.
    """
    def __init__(self, **kwargs):
        super(ConvNet, self).__init__(**kwargs)
        
        self.l1 = ConvBlock(32, (3, 3), name="l1")
        self.l2 = ConvBlock(16, (3, 3), name="l2", pool_size=(2, 2))
        self.l3 = ConvBlock(8, (3, 3), name="l3")
        self.l4 = ConvBlock(1, (3, 3), name="l4", pool_size=(2, 2))

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