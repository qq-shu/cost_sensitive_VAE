'''
Example of VAE on MNIST dataset using MLP
The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to  generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean=0 and std=1.
# Reference
[1] Kingma, Diederik P., and Max Welling.
"Auto-encoding variational bayes."
https://arxiv.org/abs/1312.6114

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix as con
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

# network parameters
batch_size = 100  # 分批，每一批大小为100
epochs = 100
original_dim = 94  # 输入94维
intermediate_dim1 = 74  # 隐藏层1
intermediate_dim2 = 32  # 隐藏层2
latent_dim = 2 # 隐变量2维
out1_dim = 94  # 重构x
out2_dim = 3   # 分类y

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps

def sampling(args):
    """Reparameterization trick by sampling
        fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    # K is the keras backend
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# VAE model = encoder + decoder
# build encoder model

x = Input(shape=(original_dim,), name='input') # 维数为original_dim（94），数量不确定
h1 = Dense(intermediate_dim1, activation='relu')(x)
h2 = Dense(intermediate_dim2, activation='relu')(h1)

# 算p(Z|X)的均值和方差
z_mean = Dense(latent_dim, name='z_mean')(h2)  # 隐层均值，输入为h，输出维数为2
z_log_var = Dense(latent_dim, name='z_log_var')(h2)  # 隐层方差，输入为h，输出维数为2


# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary
# with the TensorFlow backend
z = Lambda(sampling,
           output_shape=(latent_dim,),
           name='z')([z_mean, z_log_var])

# instantiate encoder model
# encoder = Model(x, [z_mean, z_log_var, z], name='encoder')
# encoder.summary()
# plot_model(encoder,
#            to_file='vae_mlp_encoder.png',
#            show_shapes=True)

# np.array([z_mean, z_log_var, z]).shape
# z_mean.shape
# z_log_var.shape
# z.shape

# build decoder model

# latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
decoder_h1 = Dense(intermediate_dim2, activation='relu')  # 32个神经元
decoder_h2 = Dense(intermediate_dim1, activation='relu')  # 74个神经元
decoder_mean = Dense(out1_dim, activation='sigmoid', name='output1')  # 94个神经元
decoder_y = Dense(out2_dim, activation='softmax', name='output2')

# h1_decoded = decoder_h1(z)
h1_decoded = decoder_h1(z)
h2_decoded = decoder_h2(h1_decoded)
x_decoded = decoder_mean(h2_decoded)
y_clf = decoder_y(h2_decoded)

# instantiate decoder model
# decoder = Model(latent_inputs, [x_decoded, y_clf], name='decoder')
# decoder.summary()
# plot_model(decoder,
#            to_file='vae_mlp_decoder.png',
#            show_shapes=True)

# np.array([x_decoded, y_clf]).shape
# x_decoded.shape
# y_clf.shape

a=50
L=20
M=a*L
H=a*a*L
cost=[[1,M,L],
     [M,1,H],
     [L,H,1]]
def csvae_loss_function(y_true, y_pred):
    y_true1 = y_true[0]
    y_true2 = y_true[1]
    y_pred1 = y_pred[0]
    y_pred2 = y_pred[1]
    print(y_true[0])
    print(y_true[1])
    print(y_pred[0])
    print(y_pred[1])
    reconstruction_loss = K.sum(K.binary_crossentropy(tf.convert_to_tensor(y_true1), tf.convert_to_tensor(y_pred1)), axis=-1) # 重构误差
#     reconstruction_loss = binary_crossentropy(y_true1, y_pred1)
#     reconstruction_loss = mse(x, x_deecoded)
#     reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
#     vae_loss = K.mean(reconstruction_loss + kl_loss)
    clf_loss = -tf.reduce_mean(tf.convert_to_tensor(y_true2) * tf.log(tf.convert_to_tensor(y_pred2))) * cost[np.argmax(y_true2)][np.argmax(y_pred2)]
    vae_loss = K.mean(reconstruction_loss + kl_loss) + 500 * clf_loss
    return vae_loss



# def loss1(x, x_decoded):
#     xent_loss = K.sum(K.binary_crossentropy(tf.convert_to_tensor(x), tf.convert_to_tensor(x_decoded)), axis=-1) # 重构误差
#     kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1) # KL散度
#     vae_loss = xent_loss + kl_loss
#     return vae_loss

# def loss2(y_true, y_pred):
#     clf_loss = -tf.reduce_mean(tf.convert_to_tensor(y_true) * tf.log(tf.convert_to_tensor(y_pred))) * cost[np.argmax(y_true)][np.argmax(y_pred)]
# #     clf_loss = -tf.reduce_mean(np.argmax(y_true) * tf.log(np.argmax(y_pred)) * cost[np.argmax(y_true)][np.argmax(y_pred)]
#     return clf_loss


if __name__ == '__main__':

    data = pd.read_csv("Full_Dataset-Dmax-TTT.csv")
    # 打乱数据顺序
    data = shuffle(data)
    # shuffle(data)
    # print(data.iloc[0:1,0:95])
    X = data.iloc[0:5935, 0:94]
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)
    Y = pd.get_dummies(data['Phase Formation'])
    Y = Y.values
    # type(X)
    # type(Y)
    # X.shape
    # Y.shape
    # Z = np.hstack((X, Y))
    # Z.shape
    # Y = np.hstack((X, Y))


    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    # instantiate VAE model
    # outputs = decoder(encoder(x)[2])
    vae = Model(inputs=x, outputs=[x_decoded, y_clf])
    # vae.summary()
    vae.compile(optimizer='adam',
                loss=csvae_loss_function)

    vae.fit({'input': x_train},
            {'output1': x_train, 'output2': y_train},
            shuffle=True, epochs=epochs, verbose=1, batch_size=batch_size)
    y1, y2 = vae.predict(x_test)

    y_predict = []
    for i in y2:
        y_predict.append(np.argmax(i))
    #     print(y_predict)

    y_true = []
    for j in y_test:
        y_true.append(np.argmax(j))
    #     print(y_true)

    print('混淆矩阵：')
    print(con(y_true, y_predict))

    print('Accuracy:')                                                                                                           
    print(accuracy_score(y_true, y_predict))