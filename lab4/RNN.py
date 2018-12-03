import os
import numpy as np
from conll_dictorizer import CoNLLDictorizer, Token
import datasets.py as ds
import sys
from sklearn.feature_extraction import DictVectorizer
import time
from keras import models, layers
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.models import load_model
import math
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import LSTM, Bidirectional, SimpleRNN, Dense


trace = True
path = str(open(path.conf, 'r').read()).strip()

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

def build_sequences(corpus_dict, key_x='form', key_y='upos', tolower=True):
    """
    Creates sequences from a list of dictionaries
    :param corpus_dict:
    :param key_x:
    :param key_y:
    :return:
    """
    X = []
    Y = []
    for sentence in corpus_dict:
        x = []
        y = []
        for word in sentence:
            x += [word[key_x]]
            y += [word[key_y]]
        if tolower:
            x = list(map(str.lower, x))
        X += [x]
        Y += [y]
    return X, Y

def to_index(X, idx):
    """
    Convert the word lists (or POS lists) to indexes
    :param X: List of word (or POS) lists
    :param idx: word to number dictionary
    :return:
    """
    X_idx = []
    for x in X:
        # We map the unknown words to one
        x_idx = list(map(lambda x: idx.get(x, 1), x))
        X_idx += [x_idx]
    return X_idx

def predict_sentence(sentence, model, word_idx, 
                     vocabulary_words, rev_idx_pos, verbose=False):
    # Predict one sentence
    sentence = sentence.split()
    word_idxs = to_index([sentence], word_idx)
    word_idx_padded = pad_sequences(word_idxs)

    pos_idx_pred = model.predict(word_idx_padded)
    # We remove padding
    pos_idx_pred = pos_idx_pred[0][-len(sentence):]
    pos_idx = list(map(np.argmax, pos_idx_pred))
    pos = list(map(rev_idx_pos.get, pos_idx))
    if verbose:
        print('Sentence', sentence)
        print('Sentence word indexes', word_idxs)
        print('Padded sentence', word_idx_padded)
        print('POS predicted', pos_idx_pred[0])
        print('POS shape', pos_idx_pred.shape)
    return pos

train_sentances, dev_sentances, test_sentances, column_names = ds.load_conll2003_en()
conll_dict = CoNLLDictorizer(column_names, col_sep='\t')
train_dict = conll_dict.transform(train_sentences)
dev_dict = conll_dict.transform(dev_sentences)
test_dict = conll_dict.transform(test_sentences)

embeddings_file = os.path.join(path, "corpus/glove.6B.100d.txt")
embeddings_dict = load(embedding_file)

X_train_cat, Y_train_cat = build_sequences(train_dict) # Xfor words, Y for parts of speech (pos)

vocabulary_words = sorted(list( set([word for sentence in X_train_cat for word in sentence])))
pos = sorted(list(set([pos for sentence in Y_train_cat for pos in sentence])))

NB_CLASSES = len(pos)

embeddings_words = embeddings_dict.keys()
if(trace):
    print('Words in GloVe:',  len(embeddings_dict.keys()))
vocabulary_words = sorted(list(set(vocabulary_words + list(embeddings_words))))
cnt_uniq = len(vocabulary_words) + 2
if(trace):
    print('# unique words in the vocabulary: embeddings and corpus:', cnt_uniq)

# We start at one to make provision for the padding symbol 0 
# in RNN and LSTMs and 1 for the unknown words
rev_word_idx = dict(enumerate(vocabulary_words, start=2))
rev_pos_idx = dict(enumerate(pos, start=2))
word_idx = {v: k for k, v in rev_word_idx.items()}
pos_idx = {v: k for k, v in rev_pos_idx.items()}
print('word index:', list(word_idx.items())[:10])
print('POS index:', list(pos_idx.items())[:10])

# We create the parallel sequences of indexes
X_idx = to_index(X_train_cat, word_idx)
Y_idx = to_index(Y_train_cat, pos_idx)
print('First sentences, word indices', X_idx[:3])
print('First sentences, POS indices', Y_idx[:3])

X = pad_sequences(X_idx)
Y = pad_sequences(Y_idx)

print(X[0])
print(Y[0])

# The number of POS classes and 0 (padding symbol)
Y_train = to_categorical(Y, num_classes=len(pos) + 2)
print(Y_train[0])

rdstate = np.random.RandomState(1234567)
embedding_matrix = rdstate.uniform(-0.05, 0.05, (len(vocabulary_words) + 2, EMBEDDING_DIM))
for word in vocabulary_words:
    if word in embeddings_dict:
        # If the words are in the embeddings, we fill them with a value
        embedding_matrix[word_idx[word]] = embeddings_dict[word]
if(trace):
    print('Shape of embedding matrix:', embedding_matrix.shape)
    print('Embedding of table', embedding_matrix[word_idx['table']])
    print('Embedding of the padding symbol, idx 0, random numbers', embedding_matrix[0])

model = models.Sequential()
model.add(layers.Embedding(len(vocabulary_words) + 2, EMBEDDING_DIM, mask_zero=True, input_length=None))
model.layers[0].set_weights([embedding_matrix])
# The default is True
model.layers[0].trainable = True
model.add(SimpleRNN(100, return_sequences=True))
# model.add(Bidirectional(SimpleRNN(100, return_sequences=True)))
# model.add(Bidirectional(LSTM(100, return_sequences=True)))
model.add(Dense(NB_CLASSES + 2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
model.summary()
model.fit(X, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)



# In X_dict, we replace the words with their index
X_test_cat, Y_test_cat = build_sequences(test_dict)
# We create the parallel sequences of indexes
X_test_idx = to_index(X_test_cat, word_idx)
Y_test_idx = to_index(Y_test_cat, pos_idx)

if(trace):
    print('X[0] test idx', X_test_idx[0])
    print('Y[0] test idx', Y_test_idx[0])

X_test_padded = pad_sequences(X_test_idx)
Y_test_padded = pad_sequences(Y_test_idx)
if(trace):
    print('X[0] test idx passed', X_test_padded[0])
    print('Y[0] test idx padded', Y_test_padded[0])
# One extra symbol for 0 (padding)
Y_test_padded_vectorized = to_categorical(Y_test_padded, 
                                          num_classes=len(pos) + 2)
if(trace):
    print('Y[0] test idx padded vectorized', Y_test_padded_vectorized[0])
    print(X_test_padded.shape)
    print(Y_test_padded_vectorized.shape)

# Evaluates with the padding symbol
test_loss, test_acc = model.evaluate(X_test_padded, 
                                     Y_test_padded_vectorized)
if(trace):
    print('Loss:', test_loss)
    print('Accuracy:', test_acc)

corpus_pos_predictions = model.predict(X_test_padded)pos_pred_num = []
for sent_nbr, sent_pos_predictions in enumerate(corpus_pos_predictions):
    pos_pred_num += [sent_pos_predictions[-len(X_test_cat[sent_nbr]):]]
print(pos_pred_num[:2])

pos_pred = []
for sentence in pos_pred_num:
    pos_pred_idx = list(map(np.argmax, sentence))
    pos_pred_cat = list(map(rev_pos_idx.get, pos_pred_idx))
    pos_pred += [pos_pred_cat]

print(pos_pred[:2])
print(Y_test_cat[:2])

total, correct, total_ukn, correct_ukn = 0, 0, 0, 0
for id_s, sentence in enumerate(X_test_cat):
    for id_w, word in enumerate(sentence):
        total += 1
        if pos_pred[id_s][id_w] == Y_test_cat[id_s][id_w]:
            correct += 1
        # The word is not in the dictionary
        if word not in word_idx:
            total_ukn += 1
            if pos_pred[id_s][id_w] == Y_test_cat[id_s][id_w]:
                correct_ukn += 1

print('total %d, correct %d, accuracy %f' % 
      (total, correct, correct / total))
if total_ukn != 0:
    print('total unknown %d, correct %d, accuracy %f' % 
          (total_ukn, correct_ukn, correct_ukn / total_ukn))

sentences = ["That round table might collapse .", "The man can learn well .", "The man can swim .", "The man can simwo ."]
for sentence in sentences:
    y_test_pred_cat = predict_sentence(sentence.lower(), model, word_idx, vocabulary_words, rev_pos_idx, trace)
    print(sentence)
    print(y_test_pred_cat)
