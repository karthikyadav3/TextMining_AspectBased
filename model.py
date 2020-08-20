# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 11:55:47 2019

@author: Sai Karthik Yadav
"""

import codecs
import math
import numpy as np
import preprocess
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import confusion_matrix
import seaborn as sns
class Model:
    def __init__(self, batch_size, max_seq_len, sentiment_NB,  Sentiment_NB_Word, embedding_size,   NB_Words ,
        lstm_nb_inside, layers, epochs, LEARNING_RATE, WEIGHT_DECAY, flag_train, flag_use_sentiment_for_word, session):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.sentiment_NB = sentiment_NB
        self. Sentiment_NB_Word =  Sentiment_NB_Word
        self.embedding_size = embedding_size
        self.NB_Words  = NB_Words 
        self.  lstm_nb_inside =   lstm_nb_inside
        self.layers = layers
        self.epochs = epochs
        self.LEARNING_RATE = LEARNING_RATE
        self.WEIGHT_DECAY = WEIGHT_DECAY
        self.flag_train = flag_train
        self.flag_use_sentiment_for_word = flag_use_sentiment_for_word
        self.session = session

    def modeling(self):
        self.tf_X_train = tf.placeholder(tf.float32, \
            shape = [None, self.max_seq_len, self.embedding_size - int(self.flag_use_sentiment_for_word) * self. Sentiment_NB_Word])
        self.tf_X_sent_for_word = tf.placeholder(tf.int64, shape = [None, self.max_seq_len])
        self.tf_X_train_mask = tf.placeholder(tf.float32, shape = [None, self.max_seq_len])
        self.tf_X_binary_mask = tf.placeholder(tf.float32, shape = [None, self.max_seq_len])
        self.tf_y_train = tf.placeholder(tf.int64, shape = [None, self.max_seq_len])
        self.tf_X_seq_len = tf.placeholder(tf.int64, shape = [None])
        self.keep_prob = tf.placeholder(tf.float32)
        

        self.ln_w = tf.Variable(tf.truncated_normal([self.embedding_size, self.  NB_Words ], stddev = math.sqrt(3.0 / (self.embedding_size + self.  NB_Words ))))
        self.ln_b = tf.Variable(tf.zeros([self.  NB_Words ]))
         
        self.sent_w = tf.Variable(tf.truncated_normal([self.  lstm_nb_inside, self.sentiment_NB],
                                                 stddev = math.sqrt(3.0 / self.  lstm_nb_inside + self.sentiment_NB)))
        self.sent_b = tf.Variable(tf.zeros([self.sentiment_NB]))

        y_labels = tf.one_hot(self.tf_y_train,
                              self.sentiment_NB,
                              on_value = 1.0,
                              off_value = 0.0,
                              axis = -1)
         

        if (self.flag_use_sentiment_for_word):
            X_sent_for_word = tf.one_hot(self.tf_X_sent_for_word, self. Sentiment_NB_Word,
                                 on_value = 20.0,
                                 off_value = 10.0,
                                 axis = -1)

            X_train = tf.concat([self.tf_X_train, X_sent_for_word], 2)
            X_train = tf.transpose(X_train, [1, 0, 2])
        else:
            X_train = tf.transpose(self.tf_X_train, [1, 0, 2])

        # Reshaping to (n_steps * batch_size, n_input)
        X_train = tf.reshape(X_train, [-1, self.embedding_size])
        X_train = tf.nn.relu(tf.add(tf.matmul(X_train, self.ln_w), self.ln_b))
        X_train = tf.nn.dropout(X_train, self.keep_prob)
        X_train = tf.split(axis = 0, num_or_size_splits = self.max_seq_len, value = X_train)
        
        # bidirection lstm
        # Creating the forward and backwards cells
        lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.  lstm_nb_inside, forget_bias = 0.8)
        lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.  lstm_nb_inside, forget_bias = 0.8)
        # Get lstm cell output
        outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell,
                                                     lstm_bw_cell,
                                                     X_train,
                                                     dtype='float32')

        output_fw, output_bw = tf.split(outputs, [self.  lstm_nb_inside, self.  lstm_nb_inside], 2)
        sentiment = tf.reshape(tf.add(output_fw, output_bw), [-1, self.  lstm_nb_inside])
        
      
        sentiment = tf.nn.dropout(sentiment, self.keep_prob)
        sentiment = tf.add(tf.matmul(sentiment, self.sent_w), self.sent_b)
        sentiment = tf.split(axis = 0, num_or_size_splits = self.max_seq_len, value = sentiment)

        # Change back dimension to [batch_size, n_step, n_input]
        sentiment = tf.stack(sentiment)
        sentiment = tf.transpose(sentiment, [1, 0, 2])
        sentiment = tf.multiply(sentiment, tf.expand_dims(self.tf_X_binary_mask, 2))

        self.cross_entropy = tf.reduce_mean(tf.multiply(tf.nn.softmax_cross_entropy_with_logits(logits = sentiment, labels = y_labels), self.tf_X_train_mask))
        
        regularization = self.WEIGHT_DECAY * sum(
            tf.nn.l2_loss(tf_var)
                for tf_var in tf.compat.v1.trainable_variables()
                if not ("noreg" in tf_var.name or "Bias" in tf_var.name)
        )
        self.cross_entropy = self.cross_entropy + regularization
        
        self.prediction = tf.argmax(tf.nn.softmax(sentiment), 2)
        self.correct_prediction = tf.reduce_sum(tf.multiply(tf.cast(tf.equal(self.prediction, self.tf_y_train), tf.float32), self.tf_X_binary_mask))
        self.global_step = tf.Variable(0, trainable = True)
        self.learning_rate = tf.train.exponential_decay(self.LEARNING_RATE, self.global_step, 1000, 0.65, staircase = True)
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cross_entropy, global_step = self.global_step)
        self.optimizer = tf.compat.v1.train.AdagradOptimizer(self.learning_rate).minimize(self.cross_entropy)

        self.saver = tf.train.Saver()
        self.init = tf.global_variables_initializer()


    def predict(self, data):
        prediction_test = self.session.run(self.prediction, 
                          feed_dict={self.tf_X_train: np.asarray(data.testX),
                                     self.tf_X_binary_mask: np.asarray(data.test_binary_mask),
                                     self.tf_X_seq_len: np.asarray(data.test_seq_len),
                                     self.tf_X_sent_for_word: np.asarray(data.test_sentiment_for_word),
                                     self.keep_prob: 1.0})


        ret = list()
        for i in range(len(data.test_data)):
                data_sample = ''
                for j in range(len(data.test_data[i])):
                    if data.word_dict_rev[data.test_data[i][j]] == '<unk>':
                        continue
                    elif data.test_binary_mask[i][j] > 0.:
                        data_sample = data_sample + data.word_dict_rev[data.test_data[i][j]] + \
                         '(predict ' + str(prediction_test[i][j]) + ') '
                    else:
                        data_sample = data_sample + data.word_dict_rev[data.test_data[i][j]] + ' '
                ret.append(data_sample.replace('<padding>', '').strip())
        return ret
