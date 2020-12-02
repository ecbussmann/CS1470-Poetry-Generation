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

        print(total_batch_loss)

    return None


def generate_sentence(sentence, vocab, model, sample_n=10):
    """
    Takes a model, vocab, selects from the most likely next word from the model's distribution

    :param model: trained RNN model
    :param dict: dictionary, word to id mapping
    :return: None
    """

    reverse_vocab = {idx: word for word, idx in vocab.items()}
    previous_state = None

    sentence_ids = []

    for word in sentence:
        word_id = vocab["*UNK*"]

        if word in vocab:
            word_id = vocab[word]

        sentence_ids.append(word_id)

    pad_token = vocab["*PAD*"]
    stop_token = vocab["*STOP*"]
    decoder_input = [pad_token] * 19 + [stop_token]

    sentence_logits = model.call([sentence_ids], [decoder_input])
    sentence_logits = tf.squeeze(sentence_logits)
    output_sentece = []

    for word_logits in sentence_logits:
        out_index = np.argsort(word_logits)[len(word_logits) - 1]
        output_sentece.append(reverse_vocab[out_index])

    print(" ".join(output_sentece))



def main():
    inputs, labels, padding_index, dict = get_data(
        "../../data/wiki.train.tokens")

    model = Model(len(dict))
    train(model, inputs, labels, padding_index)

    sentence = ["He", "wrote", "eighteen", "poems", "on", "painting", "alone", "more", "than", "any", "other", "Tang", "poet"]

    generate_sentence(sentence, dict, model)


if __name__ == '__main__':
    main()
