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
        #hidden_dec_expanded = tf.expand_dims(hidden_dec, 1)
        hidden_dec_transposed = tf.transpose(hidden_dec)
        dense_output = self.dense_layer(hidden_enc)
        dense_output = tf.reshape(dense_output, (dense_output.shape[0], -1))
        score = tf.matmul(hidden_dec_transposed, dense_output)
        score = tf.reshape(score, (-1, 20, 1))
        score = tf.squeeze(score)

        print("score shape")
        print(score.shape)

        attention_weights = tf.nn.softmax(score, axis=1) # axis ????????????????????????????????????

        context = tf.matmul(attention_weights, hidden_enc) # [150,20,150], [100,20,150].
        context = tf.reduce_sum(context, axis=1)

        print("context vector shape")
        print(context.shape)

        return context
