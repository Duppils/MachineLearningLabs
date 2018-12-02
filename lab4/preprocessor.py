import sys
import os
from sklearn.feature_extraction import DictVectorizer
import time
from keras import models, layers
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.models import load_model
import math


OPTIMIZER = 'rmsprop'
SCALER = True
SIMPLE_MODEL = False
BATCH_SIZE = 128
EPOCHS = 2
MINI_CORPUS = False

def load(file):
    """
    Return the embeddings in the from of a dictionary
    :param file:
    :return:
    """
    file = file
    embeddings = {}
    glove = open(file)
    for line in glove:
        values = line.strip().split()
        word = values[0]
        vector = np.array(values[1:], dtype='float32')
        embeddings[word] = vector
    glove.close()
    embeddings_dict = embeddings
    embedded_words = sorted(list(embeddings_dict.keys()))
    return embeddings_dict

path = str(open("path.conf", "r").read()).rstrip('\n')

embedding_file = os.path.join(path, 'corpus/glove.6B.100d.txt')
embeddings_dict = load(embedding_file)

def load_ud_en_ewt():

    train_file = '../datasets/ud_en/en_ewt-ud-train.conllu'
    dev_file = '../datasets/ud_en/en_ewt-ud-dev.conllu'
    test_file = '../datasets/ud_en/en_ewt-ud-test.conllu'
    column_names = ['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS', 
                    'FEATS', 'HEAD', 'DEPREL', 'HEAD', 'DEPS', 'MISC']
    column_names = list(map(str.lower, column_names))
    train_sentences = open(train_file).read().strip()
    dev_sentences = open(dev_file).read().strip()
    test_sentences = open(test_file).read().strip()
    # test2_sentences = open(test2_file).read().strip()
    return train_sentences, dev_sentences, test_sentences, column_names

def load_conll2009_pos():
    train_file = '/Users/pierre/Documents/Cours/EDAN20/corpus/conll2009/en/CoNLL2009-ST-English-train-pos.txt'
    dev_file = '/Users/pierre/Documents/Cours/EDAN20/corpus/conll2009/en/CoNLL2009-ST-English-development-pos.txt'
    test_file = '/Users/pierre/Documents/Cours/EDAN20/corpus/conll2009/en/CoNLL2009-ST-test-words-pos.txt'
    # test2_file = 'simple_pos_test.txt'

    column_names = ['id', 'form', 'lemma', 'plemma', 'pos', 'ppos']

    train_sentences = open(train_file).read().strip()
    dev_sentences = open(dev_file).read().strip()
    test_sentences = open(test_file).read().strip()
    # test2_sentences = open(test2_file).read().strip()
    return train_sentences, dev_sentences, test_sentences, column_names

# train_sentences, dev_sentences, test_sentences, column_names = \
# load_conll2009_pos()
train_sentences, dev_sentences, test_sentences, column_names =\
load_ud_en_ewt()
train_sentences[:100]
