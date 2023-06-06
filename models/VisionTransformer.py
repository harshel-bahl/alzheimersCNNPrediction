"""
This file implements a Vision Transformer (ViT) from Dosovitskiy et al. (2020).
The implementation has been adapted from https://www.kaggle.com/code/utkarshsaxenadn/vit-vision-transformer-in-keras-tensorflow.
"""

import tensorflow as tf

from .PretrainedConvNet import DenseLayer

class PatchGenerator(tf.keras.layers.Layer):

    def __init__(self, patch_size):
        super(PatchGenerator, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1,1,1,1],
            padding="VALID"
        )
        patch_dims = tf.shape(patches)[-1]
        patches = tf.reshape(patches, [tf.shape(images)[0], tf.shape(patches)[1] * tf.shape(patches)[2], patch_dims])
        return patches

class PatchEncoder(tf.keras.layers.Layer):

    def __init__(self, num_patches, projection_dims):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.d = projection_dims

        self.dense = tf.keras.layers.Dense(units=projection_dims)
        self.positional_embeddings = tf.keras.layers.Embedding(input_dim=num_patches, output_dim=projection_dims)

    def call(self, X):
        positions = tf.range(0,limit=self.num_patches, delta=1)
        encoded = self.dense(X) + self.positional_embeddings(positions)
        return encoded

class MLP(tf.keras.layers.Layer):
    def __init__(self, n_layers: int, units: int, activation: tf.keras.layers.Layer = tf.keras.layers.LeakyReLU, dropout_rate: float = 0.3):
        super(MLP, self).__init__()
        self.n_layers = n_layers
        self.units = units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.layers = [DenseLayer(units, activation=activation, dropout_rate=dropout_rate) for _ in range(n_layers)]

    def call(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Transformer(tf.keras.layers.Layer):
    def __init__(self, L=5, num_heads=1, key_dim=64, hidden_units=64):
        super(Transformer, self).__init__()
        self.L = L
        
        self.heads = num_heads
        self.key_dim = key_dim
        self.hidden_units = hidden_units

        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.MHA = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=0.1)
        self.net = MLP(n_layers=3, units=hidden_units)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        x = inputs
        for _ in range(self.L):
          x = self.norm(x)
          x = self.MHA(x, x)
          y = x + inputs
          x = self.norm(y)
          x = self.net(x)
          x = x + y
        return x
    
class ViT(tf.keras.models.Model):
    """
    Image classifier based on the Vision Transformer (ViT) architecture. 
    """
    def __init__(self):
        super(ViT, self).__init__()
        self.patch_generator = PatchGenerator(16)
        self.patch_encoder = PatchEncoder(64, 64)
        self.transformer = Transformer(8, num_heads=4, key_dim=64, hidden_units=64)
        self.layernorm =  tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.flatten = tf.keras.layers.Flatten()
        self.classifier = tf.keras.Sequential([
            DenseLayer(1024),
            DenseLayer(256),
            DenseLayer(64),
            DenseLayer(4, activation=tf.keras.layers.Softmax, dropout_rate=0.)
        ])

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        out = self.patch_generator(inputs)
        out = self.patch_encoder(out)
        out = self.transformer(out)
        out = self.layernorm(out)
        out = self.flatten(out)
        out = self.classifier(out)
        return out