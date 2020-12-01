import numpy as np
import tensorflow as tf
import re

#from attenvis import AttentionVis
#av = AttentionVis()

PAD_TOKEN = "*PAD*"
STOP_TOKEN = "*STOP*"
START_TOKEN = "*START*"
UNK_TOKEN = "*UNK*"
WINDOW_SIZE = 20


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
        padded_INPUTS = line[:WINDOW_SIZE]
        padded_INPUTS += [STOP_TOKEN] + [PAD_TOKEN] * (WINDOW_SIZE - len(padded_INPUTS)-1)
        INPUT_padded_sentences.append(padded_INPUTS)

    OUTPUT_padded_sentences = []
    for line in labels:
        padded_OUTPUTS = line[:WINDOW_SIZE]
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
    DO NOT CHANGE

  Load text data from file

    :param file_name:  string, name of data file
    :return: list of sentences, each a list of words split on whitespace
  """
    text = []
    with open(file_name, 'rt', encoding='latin') as data_file:
        text = data_file.read().strip()
        text = text.lower()
        text = re.sub('\n', '', text)
        text = re.sub('= .* =', '', text)
        text = re.split('. | ! | ? ', text)

        sentences = []
        for sentence in text:
            sentences.append(sentence.split())

    return sentences

#@av.get_data_func
def get_data(wiki_file):
    new_list = []
    sentences = read_data(wiki_file)
    for x in sentences:
        x2 = x.copy()
        x2.reverse()
        new_list.append(x2)


    inputs = sentences[:-1]
    labels = new_list[1:]

    inputs_padded, labels_padded = pad_corpus(inputs, labels)
    dict, pad = build_vocab(inputs_padded)

    inputs = convert_to_id(dict, inputs_padded)
    labels = convert_to_id(dict, labels_padded)

    return inputs, labels, pad, dict
