import numpy as np
import tensorflow as tf

from model import Model
from preprocess import get_data


def train(model, train_input, train_output, padding_index):
    num_sentences = len(train_input)
    start_index = 0

    # Iterate over the training inputs in model.batch_size increments
    while (start_index <= num_sentences - model.batch_size):
        encoder_batch = train_input[start_index: start_index + model.batch_size]
        decoder_batch = train_output[start_index: start_index + model.batch_size]

        with tf.GradientTape() as tape:
            probabilities = model.call(encoder_batch, decoder_batch[:, :-1])
            labels_batch = decoder_batch[:, 1:]
            mask = np.where(labels_batch == padding_index, 0, 1)
            total_batch_loss = model.loss_function(
                probabilities, labels_batch, mask)

        gradients = tape.gradient(total_batch_loss, model.trainable_variables)
        model.optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))

        start_index += model.batch_size

    return None



def main():
    inputs, labels, padding_index, dict = get_data(
        "../../data/wiki.train.tokens")

    model = Model(len(dict))
    train(model, inputs, labels, padding_index)


if __name__ == '__main__':
    main()
