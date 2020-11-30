import numpy as np
import tensorflow as tf

class Attention(tf.keras.layers.Layer):
    def __init__(self, attention_size):
        super(Attention, self).__init__()

        self.attention_size = attention_size

        self.dense_layer = tf.keras.layers.Dense(
            self.attention_size, activation=None, dtype=tf.float32)


    def call(self, hidden_dec, hidden_enc):
        # Comment the input and output!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        hidden_dec_transposed = tf.transpose(hidden_dec)
        score = hidden_dec_transposed * self.dense_layer(hidden_enc)

        attention_weights = tf.nn.softmax(score) # axis ????????????????????????????????????

        context = attention_weights * hidden_enc
        context = tf.reduce_sum(context, axis=1)

        return context
