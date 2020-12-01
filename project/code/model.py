import numpy as np
import tensorflow as tf
from attention import Attention

class Model(tf.keras.Model):
    def __init__(self, vocab_size):
        super(Model, self).__init__()

        self.vocab_size = vocab_size

        # Define hyperparameters
        self.encoder_decoder_size = 40
        self.embed_st_dev = 0.01

        # Define batch size and optimizer/learning rate
        self.batch_size = 100
        self.embedding_size = 102
        self.learning_rate = 0.01
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate)

        # Define embeddings, encoder, decoder, and feed forward layers
        self.embeddings = tf.Variable(
            tf.random.normal([self.vocab_size, self.embedding_size],
            stddev=self.embed_st_dev))
        self.encoder = tf.keras.layers.GRU(
            self.encoder_decoder_size, return_sequences=True, return_state=True)
        self.decoder = tf.keras.layers.GRU(
            self.encoder_decoder_size, return_sequences=True, return_state=True)
        self.attention_layer = Attention(self.encoder_decoder_size)
        self.dense_layer1 = tf.keras.layers.Dense(
            self.encoder_decoder_size, activation='tanh', dtype=tf.float32)
        self.dense_layer2 = tf.keras.layers.Dense(
            self.vocab_size, activation=None, dtype=tf.float32)

    @tf.function
    def call(self, encoder_input, decoder_input):
        """
        :param encoder_input: batched ids corresponding to input sentences
        :param decoder_input: batched ids corresponding to reversed output sentences
        :return prbs: The 3d probabilities as a tensor, [batch_size x window_size x vocab_size]
        """
        # Pass encoder sentence embeddings to the encoder
        encoder_embeddings = tf.nn.embedding_lookup(
            self.embeddings, encoder_input)
        encoder_output, encoder_final_state = self.encoder(
            encoder_embeddings, initial_state=None)

        # Use general attention
        context = self.attention_layer(encoder_final_state, encoder_output)
        context_hidden_concat = tf.concat([context, encoder_final_state], axis=-1) # axis?
        hidden_with_attention = self.dense_layer1(context_hidden_concat)

        # Pass decoder embeddings and attention enhanced hidden state to decoder
        decoder_embeddings = tf.nn.embedding_lookup(
            self.embeddings, decoder_input)
        decoder_output, decoder_final_state = self.decoder(
            decoder_embeddings,
            initial_state=hidden_with_attention)

        # Apply dense layer to the decoder out to generate probabilities
        dense_outputs = self.dense_layer2(decoder_output)

        return tf.nn.softmax(dense_outputs)


    def loss_function(self, prbs, labels, mask):
        """
        Calculates the total model cross-entropy loss after one forward pass.

        :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x vocab_size]
        :param labels:  integer tensor, word prediction labels [batch_size x window_size]
        :param mask:  tensor that acts as a padding mask [batch_size x window_size]
        :return: the loss of the model as a tensor
        """

        loss = tf.keras.losses.sparse_categorical_crossentropy(
            labels, prbs, from_logits=False)
        loss_with_mask = loss * mask

        return tf.math.reduce_sum(loss_with_mask)
