#!/usr/bin/env python
# coding: utf-8

# In[6]:

import numpy as np
import cv2 
from keras.applications import ResNet50
from tensorflow import keras
from keras.models import Model
import time
from math import log, exp
import warnings
import os
import tensorflow as tf
import sys
from keras_preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, BatchNormalization, Lambda
import numpy as np
import random
from random import sample
from sklearn.preprocessing import normalize
from keras import backend as K


# Greedy

# In[70]:


def pad(alist):
    return np.array([pad_sequences([alist], maxlen=40, truncating='post')[0]])

def generate_img_feature_vector(filepath, resnet_model):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.reshape(1, 224, 224, 3)
    
    feature_vector = resnet_model.predict(img, verbose=0).reshape(1, 2048)
    
    return feature_vector

def generate_caption(img_feature_vector, model, vocab):
    caption = [1]
    next_word = None
#     img_feature_vector = np.array([img_feature_vector])
    
    while next_word != vocab['eos'] and len(caption) != 40:
        output = model.predict([img_feature_vector, pad(caption), np.zeros(shape=(1,512)), np.zeros(shape=(1,512))], verbose=0)
        next_word = np.argsort(output)[0][-1]
        caption.append(next_word)
        
    return caption

def decode_caption(encoded_caption, vocab):
    vocab_inv = vocab_inv = {v: k for k, v in vocab.items()}
    decoded_caption = []
    for word in encoded_caption:
        if word == 0:
            continue
        decoded_caption.append(vocab_inv[word])
    return " ".join(decoded_caption)

def image_caption_generator_greedy(filename, model, vocab):
    resnet_model = ResNet50(include_top=True)
    resnet_model = Model(inputs=resnet_model.input, outputs=resnet_model.layers[-2].output)
    return decode_caption(generate_caption(generate_img_feature_vector(filename, resnet_model), model, vocab), vocab)


# Beam search

# In[66]:


# beam search 
from math import log, exp

def beam_search(fv, model, beam_size, k, vocab):
    complete_captions = []
    captions_tree = [
        [([vocab['sos']], 1)]
    ]  
    for i in range(40):
        caps_to_be_expanded = captions_tree[i]
        for item in caps_to_be_expanded:
            if item[0][-1] == vocab['eos']:
                complete_captions.append(item)
        caps_to_be_expanded = sorted(filter(lambda t: t[0][-1] != vocab['eos'], caps_to_be_expanded), key=lambda t: t[1])
        caps_to_be_expanded = caps_to_be_expanded[-k:]
        candidates = []
        if len(caps_to_be_expanded) == 0:
            return
        for caption, prob in caps_to_be_expanded:
            output = model.predict([fv, pad(caption), np.zeros(shape=(1,512)), np.zeros(shape=(1,512))], verbose=0)
            # output = 2d array
            next_words = np.argsort(output)[0][-beam_size:]
            for word in next_words:
                new_caption = caption + [word]
                new_prob = (log(output[0][word])+prob)*(1/len(new_caption)**0)
                candidates.append((new_caption, new_prob))
        captions_tree.append(candidates)
    return complete_captions

def image_caption_generator_beam(filename, model, beam_size, k, vocab):
    resnet_model = ResNet50(include_top=True)
    resnet_model = Model(inputs=resnet_model.input, outputs=resnet_model.layers[-2].output)
    fv = generate_img_feature_vector(filename, resnet_model)
#     fv = np.array([fv])
    
    complete_captions = beam_search(fv, model, beam_size, k, vocab)
    
    sorted_list = sorted(complete_captions, key=lambda x: x[1])
    
    return decode_caption(sorted_list[-1][0], vocab)


# Nucleus Sampling

# In[64]:


def nucleus_sampling(softmax, p, temp):
    actual_p = 0
    x = np.argsort(softmax)
    for item in x[::-1]:
        if actual_p < p:
            actual_p = actual_p+softmax[item]
        else:
            softmax[item] = 0
    softmax = normalize([softmax], norm='l1')[0]
    softmax = apply_temp(softmax, temp)
    next_word = random.choices(list(range(0, len(softmax))), weights = softmax, k=1)[0]
    
    return next_word

def apply_temp(softmax, temperature):
    softmax = np.log(softmax) / temperature
    softmax = np.exp(softmax)
    softmax = softmax / np.sum(softmax)
    return softmax

def generate_caption_nucleus_sampling(img_feature_vector, model, vocab):
    caption = [1]
    next_word = None

    while next_word != vocab['eos'] and len(caption) != 40:
        output = model.predict([img_feature_vector, pad(caption), np.zeros(shape=(1,512)), np.zeros(shape=(1,512))], verbose=0)

        next_word = nucleus_sampling(output[0], 0.9, 0.5)

        caption.append(next_word)

    return caption

def image_caption_generator_nucleus_sampling(filename, model, vocab):
    resnet_model = ResNet50(include_top=True)
    resnet_model = Model(inputs=resnet_model.input, outputs=resnet_model.layers[-2].output)
    warnings.filterwarnings("ignore")
    fv = generate_img_feature_vector(filename, resnet_model)
#     fv = np.array([fv])
    return decode_caption(generate_caption_nucleus_sampling(fv, model, vocab), vocab)


# In[12]:

# model
def create_model():
    vocab_size = 5185+1
    max_length = 40
    unit_size = 512

    # image feature extractor model
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(512, activation='relu')(fe1)
    fe3 = BatchNormalization()(fe2)
    fe4 = Lambda(lambda x : K.expand_dims(x, axis=1))(fe3)

    # partial caption sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 512, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)  

    LSTMLayer = LSTM(512, return_state = True, dropout=0.5)

    a0 = Input(shape=(unit_size,))
    c0 = Input(shape=(unit_size,))

    a, b, c = LSTMLayer(fe4, initial_state = [a0, c0])

    A,_,_ = LSTMLayer(se2, initial_state=[b,c])

    outputs = Dense(vocab_size, activation='softmax')(A)

    # merge the two input models
    model = Model(inputs=[inputs1, inputs2, a0, c0], outputs=outputs)
    return model


if __name__ == "__main__":
    filename = "3637013_c675de7705.jpg"
    model = create_model()
    model.load_weights("final_model.h5")
    vocab = np.load("vocab.npy", allow_pickle=True).item()
    print("Nucleus Sampling: "+image_caption_generator_nucleus_sampling(filename, model, vocab))
    print("Greedy: "+image_caption_generator_greedy(filename, model, vocab))
    print("Beam: "+image_caption_generator_beam(filename, model, 5, 5, vocab))

