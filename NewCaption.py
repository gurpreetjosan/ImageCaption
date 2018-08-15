from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.layers import Input
from keras.models import load_model
from numpy import array
# extract features from each photo in the directory
def extract_features(filename):
	# load the model
	in_layer = Input(shape=(224, 224, 3))
	model = VGG16(include_top=True, input_tensor=in_layer)
	# re-structure the model
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	print(model.summary())
	# load the photo
	image = load_img(filename, target_size=(224, 224))
	# convert the image pixels to a numpy array
	image = img_to_array(image)
	# reshape data for the model
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# prepare the image for the VGG model
	image = preprocess_input(image)
	# get features
	feature = model.predict(image, verbose=0)
	return feature
 
# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# generate a description for an image
def generate_desc(model, tokenizer, photo, newstext, max_length, max_news_len):
    # seed the generation process
    in_text = 'startseq'
    # integer encode news sequence
    in_news = tokenizer.texts_to_sequences([newstext])[0]
    news_seq = pad_sequences([in_news], maxlen=max_news_len)
    # iterate over the whole length of the sequence
    for i in range(1,max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo, sequence, news_seq], verbose=0)
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
 
# load the tokenizer
tokenizer = load(open('tokenizerOriginal.pkl', 'rb'))
# pre-define the max sequence length (from training)
max_length = 50
# pre-define the max news length (from training)
max_news_length = 1636
# load the model
model = load_model('model-ep030-loss5.136-val_loss5.630.h5')
# load and prepare the photograph
photo = extract_features('/home/a/pyProjects/capt_proj/testingFiles/9.jpg')
# prepare news doc
newstext = load_doc('/home/a/pyProjects/capt_proj/testingFiles/9.txt')
# generate description
description = generate_desc(model, tokenizer, photo, newstext, max_length, max_news_length)

a, _, b = description.rpartition("startseq")
description = a + b
a, _, b = description.rpartition("endseq")
description = a + b

print('\n')
print('Caption: ',description)
