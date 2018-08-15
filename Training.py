#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 23:46:51 2018

@author: gurpreet
"""

#from os import listdir
import os
from numpy import array
from numpy import argmax
from pandas import DataFrame
from nltk.translate.bleu_score import corpus_bleu
from pickle import load
from keras.models import load_model

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Embedding,Masking
from keras.layers.merge import concatenate
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint,EarlyStopping

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r',encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# load newsdoc into memory
def load_newsdoc(filename,foldername):
    # open the file as read only
    file = open(foldername+'/'+filename, 'r', encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# load a pre-defined list of photo identifiers
def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    # process line by line
    for line in doc.split('\n'):
        # skip empty lines
        if len(line) < 1:
            continue
        # get the image identifier
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)

# split a dataset into train/test elements
def train_test_split(dataset):
    # order keys so the split is consistent
    ordered = sorted(dataset)
    end=int(len(dataset)*0.9)
    return set(ordered[:end]), set(ordered[end+1:])

# split a train dataset into train/val elements
def train_val_split(traindataset):
    # order keys so the split is consistent
    ordered = sorted(traindataset)
    end=int(len(traindataset)*0.9)
    return set(ordered[:end]), set(ordered[end+1:])

# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
    # load document
    doc = load_doc(filename)
    descriptions = dict()
    k=1
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        #print(k,'\t',line+'\n')
        # split id from description
        image_id, image_desc = tokens[0], tokens[1:]
        # skip images not in the set
        if image_id in dataset:
            # store
            descriptions[image_id] = 'startseq ' + ' '.join(image_desc) + ' endseq'
        k=k+1
    return descriptions

# load clean newstext into memory
def load_clean_newstext(foldername, dataset):
    newstext = dict()
    for filename in os.listdir(foldername):
        # load document
        doc = load_newsdoc(filename,foldername)
        image_id=filename.split('.')[0]
        if image_id in dataset:
            # store
            newstext[image_id] = doc
    return newstext

# load photo features
def load_photo_features(filename, dataset):
    # load all features
    all_features = load(open(filename, 'rb'))
    # filter features
    features = {k: all_features[k] for k in dataset}
    return features

# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions,newstext, valDesc, valNews):
    lines = list(descriptions.values())
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    lines = list(valDesc.values())
    tokenizer.fit_on_texts(lines)
    lines=list(newstext.values())
    tokenizer.fit_on_texts(lines)
    lines = list(valNews.values())
    tokenizer.fit_on_texts(lines)
    return tokenizer

# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, desc, image, news, max_length, max_lengthNews):
    Ximages, XSeq, Xtext, y = list(), list(), list(),list()
    vocab_size = len(tokenizer.word_index) + 1
    # integer encode the description
    seq = tokenizer.texts_to_sequences([desc])[0]
    news_seq=tokenizer.texts_to_sequences([news])[0]
    # split one sequence into multiple X,y pairs
    for i in range(1, len(seq)):
        # select
        in_seq, out_seq = seq[:i], seq[i]
        # pad input sequence
        in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
        news_seq=pad_sequences([news_seq],maxlen=max_lengthNews)[0]
        # encode output sequence
        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
        # store
        Ximages.append(image)
        XSeq.append(in_seq)
        Xtext.append(news_seq)
        y.append(out_seq)
    # Ximages, XSeq, y = array(Ximages), array(XSeq), array(y)
    return [Ximages, XSeq, Xtext, y]

# define the captioning model 
#this is model that combine img feature withs caption feature and pass it to LSTM layer. We have 2 LSTM layers here
# define the captioning model 
#this is model that combine img feature withs caption feature and pass it to LSTM layer. We have 2 LSTM layers here
def define_model(vocab_size, max_length, max_length_news):
    # feature extractor (encoder)
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    #fe2 = Dense(128, activation='relu')(fe1)
    fe3 = RepeatVector(max_length)(fe1)
    # embedding
    inputs2 = Input(shape=(max_length,))
    mask = Masking(mask_value=0)(inputs2)
    emb2 = Embedding(vocab_size, 200, mask_zero=True)(mask)
    emb3 = LSTM(256, return_sequences=True)(emb2)
    emb3 = Dropout(0.5)(emb3)
    emb4 = TimeDistributed(Dense(128, activation='relu'))(emb3)
    #thought vector
    src_txt_length = max_length_news
    sum_txt_length = max_length
    inputs3 = Input(shape=(src_txt_length,))
    mask = Masking(mask_value=0)(inputs3)
    encoder1 = Embedding(vocab_size, 200,mask_zero=True)(mask)
    encoder2 = LSTM(128)(encoder1)
    encoder2=Dropout(0.5)(encoder2)
    encoder3 = RepeatVector(sum_txt_length)(encoder2)

    # merge inputs
    merged = concatenate([fe3, emb4, encoder3])
    merged=Dropout(0.5)(merged)
    # language model (decoder)
    lm2 = LSTM(250)(merged)
    #lm3 = Dense(500, activation='relu')(lm2)
    outputs = Dense(vocab_size, activation='softmax')(lm2)
    # tie it together [image, seq, news] [word]
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True)
    return model


def define_model_old(vocab_size, max_length, max_length_news):
    # feature extractor (encoder)
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(128, activation='relu')(fe1)
    fe3 = RepeatVector(max_length)(fe2)
    # embedding
    inputs2 = Input(shape=(max_length,))
    emb2 = Embedding(vocab_size, max_length, mask_zero=True)(inputs2)
    emb3 = LSTM(256, return_sequences=True)(emb2)
    emb3=Dropout(0.5)(emb3)
    emb4 = TimeDistributed(Dense(128, activation='relu'))(emb3)
    #thought vector
    src_txt_length = max_length_news
    sum_txt_length = max_length
    inputs3 = Input(shape=(src_txt_length,))
    encoder1 = Embedding(vocab_size, 128)(inputs3)
    encoder2 = LSTM(128)(encoder1)
    encoder2=Dropout(0.5)(encoder2)
    encoder3 = RepeatVector(sum_txt_length)(encoder2)

    # merge inputs
    merged = concatenate([fe3, emb4, encoder3])
    merged=Dropout(0.5)(merged)
    # language model (decoder)
    lm2 = LSTM(500)(merged)
    lm3 = Dense(500, activation='relu')(lm2)
    outputs = Dense(vocab_size, activation='softmax')(lm3)
    # tie it together [image, seq, news] [word]
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True)
    return model

# This is old model where we have combined img feature with text embeddings and pass through LSTM
def define_model_1(vocab_size, max_length):
    # feature extractor model
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    # sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    # decoder model
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # summarize model
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True)
    return model

# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, features, detailnews, tokenizer, max_length, max_length_news, n_step):
    # loop until we finish training
    while 1:
        # loop over photo identifiers in the dataset
        keys = list(descriptions.keys())
        for i in range(0, len(keys), n_step):
            Ximages, XSeq, Xtext, y = list(), list(), list(),list()
            for j in range(i, min(len(keys), i+n_step)):
                image_id = keys[j]
                # retrieve photo feature input
                image = features[image_id][0]
                # retrieve text input
                desc = descriptions[image_id]
                news = detailnews[image_id]
                # generate input-output pairs
                in_img, in_seq, in_news, out_word = create_sequences(tokenizer, desc, image, news, max_length, max_length_news)
                for k in range(len(in_img)):
                    Ximages.append(in_img[k])
                    XSeq.append(in_seq[k])
                    Xtext.append(in_news[k])
                    y.append(out_word[k])
            # yield this batch of samples to the model
            yield [[array(Ximages), array(XSeq), array(Xtext)], array(y)]

# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# generate a description for an image
def generate_desc(model, tokenizer, photo, newstext, max_length, maxNewsLength):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # integer encode news sequence
        in_news = tokenizer.texts_to_sequences([newstext])[0]
        news_seq = pad_sequences([in_news], maxlen=maxNewsLength)
        # predict next word
        yhat = model.predict([photo,sequence,news_seq], verbose=0)
        # convert probability to integer
        yhat = argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text

# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, newstext, tokenizer, max_length, maxNewsLength):
    actual, predicted = list(), list()
    # step over the whole set
    for key, desc in descriptions.items():
        # generate description
        yhat = generate_desc(model, tokenizer, photos[key], newstext[key], max_length, maxNewsLength)
        # store actual and predicted
        actual.append([desc.split()])
        predicted.append(yhat.split())
    # calculate BLEU score
    bleu = corpus_bleu(actual, predicted)
    print('Bleu Score: ',bleu)
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

    return bleu

# load dev set
filename = 'trainImgs.txt'
dataset = load_set(filename)
print('Dataset: %d' % len(dataset))
# train-test split
traindata, test = train_test_split(dataset)
# train-val split
train, val=train_val_split(traindata)
# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
val_descriptions = load_clean_descriptions('descriptions.txt', val)
test_descriptions = load_clean_descriptions('descriptions.txt', test)
print('Descriptions: train=%d, val=%d, test=%d' % (len(train_descriptions), len(val_descriptions), len(test_descriptions)))
# photo features
train_features = load_photo_features('features.pkl', train)
val_features = load_photo_features('features.pkl', val)
test_features = load_photo_features('features.pkl', test)
print('Photos: train=%d, val=%d, test=%d' % (len(train_features), len(val_features), len(test_features)))
# news text
train_newstext = load_clean_newstext('newstext', train)
val_newstext = load_clean_newstext('newstext', val)
test_newstext = load_clean_newstext('newstext', test)
print('Descriptions: train=%d, val=%d, test=%d' % (len(train_newstext), len(val_newstext), len(test_newstext)))

# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions,train_newstext,val_descriptions,val_newstext)
# save the tokenizer
dump(tokenizer, open('tokenizerOriginal.pkl', 'wb'))
#print(tokenizer.word_index)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum train sequence length
max_length = max(max(len(s.split()) for s in list(train_descriptions.values())),max(len(s.split()) for s in list(val_descriptions.values())))
print('Maximum Description Length: %d' % max_length)
# determine the train maximum news text
max_length_news = max(max(len(s.split()) for s in list(train_newstext.values())),max(len(s.split()) for s in list(val_newstext.values())))
print('Maximum Newstext Length: %d' % max_length_news)

# define experiment
model_name = 'baseline1'
verbose = 1
n_epochs = 37
n_photos_per_update = 2
n_batches_per_epoch = int(len(train) / n_photos_per_update)
n_repeats = 1
validationSteps = int(len(val) / n_photos_per_update)
epoch_num=35
resume_trng= 'no'
#run experiment
train_results, test_results = list(), list()

if resume_trng =='yes':
    model = load_model ('model-ep035-loss5.122-val_loss5.619.h5')
    for i in range(n_repeats):
        # define the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print("IN..........")
        print(model.summary())
        print("OUT................")
        # define checkpoint callback
        filepath = 'EachEpoch_model_new/model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1)
        earlystop = EarlyStopping(monitor="val_loss",patience=2)
        # fit model
        model.fit_generator(
            data_generator(train_descriptions, train_features, train_newstext, tokenizer, max_length, max_length_news,
                           n_photos_per_update), validation_data=data_generator(val_descriptions, val_features, val_newstext, tokenizer, max_length,
                           max_length_news, n_photos_per_update), validation_steps=validationSteps,steps_per_epoch=n_batches_per_epoch, epochs=n_epochs,
                           verbose=verbose, callbacks=[checkpoint,earlystop], initial_epoch= epoch_num)

        # evaluate model on training data
        train_score = evaluate_model(model, train_descriptions, train_features, train_newstext, tokenizer, max_length, max_length_news)
        test_score = evaluate_model(model, test_descriptions, test_features, test_newstext, tokenizer, max_length, max_length_news)
        # store
        train_results.append(train_score)
        test_results.append(test_score)
        print('>%d: train=%f test=%f' % ((i+1), train_score, test_score))
        # save results to file
        df = DataFrame()
        df['train'] = train_results
        df['test'] = test_results
        print(df.describe())
        df.to_csv(model_name+'.csv', index=False)
else:
    # define the model
    model = define_model(vocab_size, max_length, max_length_news)
    for i in range(n_repeats):
        # define checkpoint callback
        #filepath = 'EachEpoch_model_new/model-ep{epoch:03d}-loss{loss:.3f}.h5'
        filepath = 'EachEpoch_model_new/model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1)
        earlystop=EarlyStopping(monitor="val_loss",patience=2)
        # fit model
        model.fit_generator(
            data_generator(train_descriptions, train_features, train_newstext, tokenizer, max_length, max_length_news,
                           n_photos_per_update), validation_data= data_generator(val_descriptions, val_features, val_newstext, tokenizer, max_length,
                           max_length_news, n_photos_per_update),validation_steps=validationSteps,steps_per_epoch=n_batches_per_epoch, epochs=n_epochs,
            verbose=verbose, callbacks=[checkpoint,earlystop])
        
        # evaluate model on training data
        train_score = evaluate_model(model, train_descriptions, train_features, train_newstext, tokenizer, max_length,max_length_news)
        test_score = evaluate_model(model, test_descriptions, test_features, test_newstext, tokenizer, max_length,max_length_news)
        # store
        train_results.append(train_score)
        test_results.append(test_score)
        print('>%d: train=%f test=%f' % ((i + 1), train_score, test_score))
        # save results to fil
        df = DataFrame()
        df['train'] = train_results
        df['test'] = test_results
        print(df.describe())
        df.to_csv(model_name + '.csv', index=False)
