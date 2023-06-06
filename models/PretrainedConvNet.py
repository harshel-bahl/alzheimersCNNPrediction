import tensorflow as tf


class DenseLayer(tf.keras.layers.Layer):
    """
    A fully-connected layer, with optional dropout, activation function, and residual connection.
    """
    def __init__(self, units: int, residual: bool = True,
                 activation: tf.keras.layers.Layer = tf.keras.layers.LeakyReLU,
                 dropout_rate: float = 0.5, **kwargs):
        super(DenseLayer, self).__init__(**kwargs)
        self.units = units
        self.residual = residual
        
        self.dense = tf.keras.layers.Dense(units)
        self.activation = activation()
        
        self.dropout_rate = dropout_rate
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        
        out = self.dense(inputs)
        if self.dropout_rate:
            out = self.dropout(out)
        out = self.activation(out)
        if self.residual and out.shape == inputs.shape:
            out = inputs + out
        return out


class PretrainedConvNet(tf.keras.models.Model):
    """
    A convolutional network classifier based on the EfficientNet architecture.
    """
    def __init__(self, *args, **kwargs):
        super(PretrainedConvNet, self).__init__(*args, **kwargs)
        
        self.pretrained = tf.keras.applications.efficientnet_v2.EfficientNetV2B2(weights="imagenet",
                                                                                 include_preprocessing=True, include_top=False)
        
        self.flatten = tf.keras.layers.Flatten()
        
        self.fc1 = DenseLayer(64, activation=tf.keras.layers.LeakyReLU)
        self.fc2 = DenseLayer(4, activation=tf.keras.layers.Softmax, dropout_rate=0.)
        
    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        
        out = self.pretrained(inputs)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out