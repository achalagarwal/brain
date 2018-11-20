Implementation of Paper:
https://arxiv.org/abs/1709.08820

Converting Your Thoughts to Text

-------------------------------Language-------------------------------
The code is written in Python. It is compatible with both versions 2.7 and 3.6.


-------------------------------Libraries------------------------------
Tensorflow was the library of choice for implementing the paper. The other supporing libraries are:
1) Scipy  2) NumPy 3) Scikit-learn  4)xgboost 5)random 6)sys 7)time


-------------------------------Dataset--------------------------------
The dataset used for building the model is a subset of the 'eegmmidb' dataset. 

Format: The input data is in the form of a .mat file

Data Description: The number of data points are ~29500. Each data point is a 65 vector (including one label)
The first 64 entries in the vector are the eeg signals recorded from 64 channels. The last entry is a class (1-5)
We take the first 28000 points for training our model.

Train and Test: 75% of the data is used for training and the rest is used for testing.
The train data is divided into three batches for training (each batch of 7000 point)


------------------------------Structure---------------------------------
The code contains four modules: 1)CNN 2)RNN 3)AutoEncoder 4)XGBoost
The training happens independently for the RNN and CNN part. The outputs of these two parts are
then combined. This combination serves as the input to the AutoEncoder. The output of the 
autoencoder is then fed to the XGBoost classifier.


