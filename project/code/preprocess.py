import numpy as np
import tensorflow as tf
import re

#from attenvis import AttentionVis
#av = AttentionVis()

PAD_TOKEN = "*PAD*"
STOP_TOKEN = "*STOP*"
START_TOKEN = "*START*"
UNK_TOKEN = "*UNK*"
WINDOW_SIZE = 10 #20


def pad_corpus(inputs, labels):
    """

    arguments are lists of input, label sentences. The
    text is given an initial "*STOP*".  All sentences are padded with "*STOP*" at
    the end.

    :param inputs: list of input sentences
    :param labels: list of label sentences
    :return: A tuple of: (list of padded input sentences, list of padded label sentences)
    """
    INPUT_padded_sentences = []
    for line in inputs:
        padded_INPUTS = line[:WINDOW_SIZE-1]
        padded_INPUTS += [STOP_TOKEN] + [PAD_TOKEN] * (WINDOW_SIZE - len(padded_INPUTS)-1)
        INPUT_padded_sentences.append(padded_INPUTS)

    OUTPUT_padded_sentences = []
    for line in labels:
        padded_OUTPUTS = line[:WINDOW_SIZE-1]
        padded_OUTPUTS = [START_TOKEN] + padded_OUTPUTS + [STOP_TOKEN] + [PAD_TOKEN] * (WINDOW_SIZE - len(padded_OUTPUTS)-1)
        OUTPUT_padded_sentences.append(padded_OUTPUTS)

    return INPUT_padded_sentences, OUTPUT_padded_sentences

def build_vocab(sentences):
    """


  Builds vocab from list of sentences

    :param sentences:  list of sentences, each a list of words
    :return: tuple of (dictionary: word --> unique index, pad_token_idx)
  """
    tokens = []
    for s in sentences: tokens.extend(s)
    all_words = sorted(list(set([STOP_TOKEN,PAD_TOKEN,UNK_TOKEN] + tokens)))

    vocab =  {word:i for i,word in enumerate(all_words)}

    return vocab,vocab[PAD_TOKEN]


def convert_to_id(vocab, sentences):
    """
    DO NOT CHANGE

  Convert sentences to indexed

    :param vocab:  dictionary, word --> unique index
    :param sentences:  list of lists of words, each representing padded sentence
    :return: numpy array of integers, with each row representing the word indeces in the corresponding sentences
  """
    return np.stack([[vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in sentence] for sentence in sentences])


def read_data(file_name):
    """
  Load text data from file

    :param file_name:  string, name of data file
    :return: list of sentences, each a list of words split on whitespace
  """
    text = []
    with open(file_name, 'rt', encoding='latin') as data_file:
        text = data_file.read().strip()
        text = text.lower()
        text = re.sub('\n = .* = \n', '', text)
        text = text.replace('\n', '')
        text = text.replace('!', '.')
        text = text = text.replace('?', '.')
        text = text.split('.')

        sentences = []
        for sentence in text:
            stripped_sentence = sentence.strip()

            if (stripped_sentence != ""):
                sentences.append(sentence.strip().split())

    return sentences


def get_data(train_file, test_file):
    # organizing the training data
    train_reversed_sentences = []
    train_sentences = read_data(train_file)
    for x in train_sentences:
        x2 = x.copy()
        x2.reverse()
        train_reversed_sentences.append(x2)

    train_inputs = train_sentences[:-1]
    train_labels = train_reversed_sentences[1:]

    train_inputs_padded, train_labels_padded = pad_corpus(train_inputs, train_labels)
    dict, pad = build_vocab(train_inputs_padded)
    dict[START_TOKEN] = len(dict)

    train_inputs = convert_to_id(dict, train_inputs_padded)
    train_labels = convert_to_id(dict, train_labels_padded)


    # organizing the testing data
    test_reversed_sentences = []
    test_sentences = read_data(test_file)
    for x in test_sentences:
        x2 = x.copy()
        x2.reverse()
        test_reversed_sentences.append(x2)

    test_inputs = test_sentences[:-1]
    test_labels = test_reversed_sentences[1:]

    test_inputs_padded, test_labels_padded = pad_corpus(test_inputs, test_labels)

    test_inputs = convert_to_id(dict, test_inputs_padded)
    test_labels = convert_to_id(dict, test_labels_padded)

    return train_inputs, train_labels, test_inputs, test_labels, pad, dict
