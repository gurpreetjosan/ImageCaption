# ImageCaption
Generate captions from news text and image. 

This project generate captions for images. The features of image along with news text has been trained and model tries to caption the image given news text and corresponding image. Image features are extracted using VGG model and news text features are collected using Bidirectional LSTM network.

# Data
You can use your own data. If you need our data please email us at josangurpreet@pbi.ac.in. If you use the dataset, please cite the work as follow:
Kaur, M., Josan, G., & Kaur, J. (2021). Automatic Punjabi Caption Generation For Sports Images. INFOCOMP Journal of Computer Science, 20(1). Retrieved from http://infocomp.dcc.ufla.br/index.php/infocomp/article/view/1180

# How to run
Extract Features of image beforhand using command:

python3 FeatureExtractor.py

Then train the system using command 

python3 Training.py

