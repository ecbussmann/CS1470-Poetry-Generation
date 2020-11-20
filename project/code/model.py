import numpy as np
import tensorflow as tf

class GRU_Model(tf.keras.Model):
    def __init__(self, vocab_size):
        ###### DO NOT CHANGE ##############
        super(GRU_Model, self).__init__()


        # Define hyperparameters
        self.encoder_decoder_size = 40
        self.embed_st_dev = 0.01

        # Define batch size and optimizer/learning rate
        self.batch_size = 100 # You can change this
        self.embedding_size = 100
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

    @tf.function
    def call(self, encoder_input, decoder_input):
        """
        :param encoder_input: batched ids corresponding to input sentences
        :param decoder_input: batched ids corresponding to reversed output sentences
        :return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
        """

        # NOT DONE

        # Pass encoder sentence embeddings to the encoder
        encoder_embeddings = tf.nn.embedding_lookup(
            self.embeddings, encoder_input)
        encoder_output, encoder_final_state = self.encoder(
            encoder_embeddings, initial_state=None)

        # Pass decoder embeddings and final state of the encoder to decoder
        decoder_embeddings = tf.nn.embedding_lookup(
            self.embeddings, decoder_input)
        decoder_output, decoder_final_state = self.decoder(
            decoder_embeddings,
            initial_state=encoder_final_state)

        # Apply dense layer to the decoder out to generate probabilities
        dense_outputs = self.dense(decoder_output)

        return tf.nn.softmax(dense_outputs)
