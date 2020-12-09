import numpy as np
import tensorflow as tf

from model import Model
from preprocess import get_data, pad_corpus, convert_to_id


def train(model, train_input, train_output, padding_index):
    """
    Runs through one epoch - all training examples.

    :param model: the initialized model to use for forward and backward pass
    :param train_input: input train data (all data for training) of shape
        (num_sentences, window_size)
    :param train_output: output train data (all data for training) of shape
        (num_sentences, window_size + 1)
    :param padding_index: the padding index, the id of *PAD* token. This integer
        is used when masking padding labels.
    :return: None
    """
    num_sentences = len(train_input)
    start_index = 0

    # Iterate over the training inputs in model.batch_size increments
    while (start_index <= num_sentences - model.batch_size):
        # Get batches from the inputs and outputs
        encoder_batch = train_input[start_index: start_index + model.batch_size]
        decoder_batch = train_output[start_index: start_index + model.batch_size]

        with tf.GradientTape() as tape:
            # For the decoder input, ignore the last token in each sentence
            probabilities = model.call(encoder_batch, decoder_batch[:, :-1])

            # Ignore the first token in each sentence
            labels_batch = decoder_batch[:, 1:]

            mask = np.where(labels_batch == padding_index, 0, 1)
            total_batch_loss = model.loss_function(
                probabilities, labels_batch, mask)

        gradients = tape.gradient(total_batch_loss, model.trainable_variables)
        model.optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))

        start_index += model.batch_size

    return None


def test(model, test_input, test_output, padding_index):
    """
    Runs through one epoch - all testing examples.

    :param model: the initialized model to use for forward and backward pass
    :param test_input: input test data (all data for testing) of shape
        (num_sentences, window_size)
    :param test_output: output test data (all data for testing) of shape
        (num_sentences, window_size + 1)
    :param padding_index: the padding index, the id of *PAD* token. This integer
        is used when masking padding labels.
    :returns: a tuple containing at index 0 the perplexity of the test set and
      at index 1 the per symbol accuracy on test set,
    """
    num_sentences = len(test_input)
    start_index = 0
    num_batches = 0
    loss_sum = 0
    accuracy_sum = 0
    num_words = 0

    # Iterate over the training inputs in model.batch_size increments
    while (start_index <= num_sentences - model.batch_size):
        # Batch the inputs and outputs
        encoder_batch = test_input[start_index: start_index + model.batch_size]
        decoder_batch = test_output[start_index: start_index + model.batch_size]

        probabilities = model.call(encoder_batch, decoder_batch[:, :-1])
        labels_batch = decoder_batch[:, 1:]
        mask = np.where(labels_batch == padding_index, 0, 1)
        total_batch_loss = model.loss_function(
            probabilities, labels_batch, mask)

        # Update the counters and summations
        num_words_in_batch = np.count_nonzero(mask)
        num_words += num_words_in_batch
        loss_sum += total_batch_loss
        batch_accuracy = model.accuracy_function(
            probabilities, labels_batch, mask)
        accuracy_sum += batch_accuracy * num_words_in_batch

        start_index += model.batch_size
        num_batches += 1

    perplexity = np.exp(loss_sum / num_words)
    per_symbol_accuracy = accuracy_sum / num_words

    return perplexity, per_symbol_accuracy


def generate_sentence(sentence, vocab, model, sample_n=10):
    """
    Takes a model, vocab, selects from the most likely next word from the model's distribution

    :param model: trained RNN model
    :param dict: dictionary, word to id mapping
    :return: None
    """
    reverse_vocab = {idx: word for word, idx in vocab.items()}

    # Preprocess the sentence
    sentence = sentence.strip()
    sentence = sentence.lower()
    sentence = sentence.replace('\n', '')
    sentence = sentence.replace('!', '')
    sentence = sentence.replace('?', '')
    sentence = sentence.replace('.', '')
    sentence = sentence.split()

    padded_sentence, _ = pad_corpus([sentence], [])
    sentence_ids = convert_to_id(vocab, padded_sentence)

    start_token = vocab["*START*"]
    stop_token = vocab["*STOP*"]
    decoder_input = [start_token] * 10

    sentence_logits = model.call(sentence_ids, [decoder_input])
    sentence_logits = tf.squeeze(sentence_logits)
    output_sentece = []

    for word_logits in sentence_logits:
        np_word_logits = word_logits.numpy()

        # Make it so highly likely and unhelpful words are not chosen
        np_word_logits[vocab["_s"]] = 0
        np_word_logits[stop_token] = 0

        top_n = np.argsort(np_word_logits)[-sample_n:]
        n_logits = np.exp(np_word_logits[top_n])/np.exp(np_word_logits[top_n]).sum()

        # Sample from the top n words to introduce variability
        out_index = np.random.choice(top_n,p=n_logits)
        output_sentece.append(reverse_vocab[out_index])

    print(" ".join(output_sentece))


def main():
    train_inputs, train_labels, test_inputs, test_labels, padding_index, dict =\
        get_data("../../data/train.txt", "../../data/test.txt")

    model = Model(len(dict))
    train(model, train_inputs, train_labels, padding_index)
    perplexity, accuracy = test(
        model, test_inputs, test_labels, padding_index)

    print("perplexity: ")
    print(perplexity)
    print("accuracy: ")
    print(accuracy)

    sentence = "He wrote eighteen poems on painting alone, more than any other Tang Poet."
    generate_sentence(sentence, dict, model)


if __name__ == '__main__':
    main()
