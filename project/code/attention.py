import numpy as np
import tensorflow as tf

class Attention(tf.keras.layers.Layer):
    def __init__(self, attention_size):
        super(Attention, self).__init__()

        self.attention_size = attention_size

        self.dense_layer = tf.keras.layers.Dense(
            self.attention_size, activation=None, dtype=tf.float32)


    def call(self, hidden_dec, hidden_enc):
        """
        Computes context vectors for general attention

        :param hidden_dec: the decoder's current hidden state, with shape
            (batch_size, attention_size)
        :param hidden_enc: the encoder's hidden state, with shape
            (batch_size, window_size, attention_size)
        :return: the context vector created using the given hidden states, with
            shape (batch_size, attention_size)
        """
        hidden_dec_transposed = tf.transpose(hidden_dec)
        dense_output = self.dense_layer(hidden_enc)
        dense_output = tf.reshape(dense_output, (dense_output.shape[0], -1))
        score = tf.matmul(hidden_dec_transposed, dense_output)
        score = tf.reshape(score, (-1, hidden_enc.shape[1], 1))
        score = tf.squeeze(score)

        attention_weights = tf.nn.softmax(score, axis=1)

        context = tf.matmul(attention_weights, hidden_enc)
        context = tf.reduce_sum(context, axis=1)

        return context
