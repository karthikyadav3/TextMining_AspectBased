# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 11:54:20 2019

@author: Sai Karthik Yadav
"""


import numpy as np
import preprocess
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator



class Data:
    def __init__(self, Category, Directory, label_dict, max_seq_len,
     File_Change_to_text_File, negative_weight, positive_weight, neutral_weight, Sentiment_label):
        self.max_seq_len = max_seq_len
        self.label_dict = label_dict
        self.Category = Category
        self.Directory = Directory
        self.File_Change_to_text_File = File_Change_to_text_File
        self.negative_weight = negative_weight
        self.positive_weight = positive_weight
        self.neutral_weight = neutral_weight
        self.Sentiment_label = Sentiment_label

        self.train_data, self.train_mask, self.train_binary_mask, self.train_label, self.train_seq_len, self.train_sentiment_for_word, \
        self.test_data, self.test_mask, self.test_binary_mask, self.test_label, self.test_seq_len, self.test_sentiment_for_word, \
        self.word_dict, self.word_dict_rev, self.embedding, aspect_list = preprocess.load_data(
            self.Category,
            self.Directory,
            self.label_dict,
            self.max_seq_len,
            self.File_Change_to_text_File,
            self.negative_weight,
            self.positive_weight,
            self.neutral_weight,
           self.Sentiment_label
        )

        self.NB_train = len(self.train_data)
        
        self.testX = list()
        for i in range(len(self.test_data)):
            sentence = list()
            for word_id in self.test_data[i]:
                sentence.append(self.embedding[word_id])
            self.testX.append(sentence)
            
        self.x_train = list()
        for i in range(len(self.train_data)):
            sentence = list()
            for word_id in self.train_data[i]:
                sentence.append(self.embedding[word_id])
            self.x_train.append(sentence)
            
def train(self, data):
        self.session.run(self.init)
        #self.load_model()
        loss_list = list()
        accuracy_list = list()

        for it in range(self.epochs):
            #generate batch (x_train, y_train, seq_lengths_train)
            if (it * self.batch_size % data.NB_train + self.batch_size < data.NB_train):
                index = it * self.batch_size % data.NB_train
            else:
                index = data.NB_train - self.batch_size
            

            self.session.run(self.optimizer, 
                          feed_dict={self.tf_X_train: np.asarray(data.x_train[index : index + self.batch_size]),
                                     self.tf_X_train_mask: np.asarray(data.train_mask[index : index + self.batch_size]),
                                     self.tf_X_binary_mask: np.asarray(data.train_binary_mask[index : index + self.batch_size]),
                                     self.tf_X_seq_len: np.asarray(data.train_seq_len[index : index + self.batch_size]),
                                     self.tf_X_sent_for_word: np.asarray(data.train_sentiment_for_word[index : index + self.batch_size]),
                                     self.tf_y_train: np.asarray(data.train_label[index : index + self.batch_size]),
                                     self.keep_prob: 0.5})

            if it % (len(data.x_train) // self.batch_size) == 0:
                print(it)
                self.evaluate(data, it + 100 >= self.epochs, self.flag_train)
                
                correct_prediction_train, cost_train = self.session.run([self.correct_prediction, self.cross_entropy], 
                                                  feed_dict={self.tf_X_train: np.asarray(data.x_train),
                                                             self.tf_X_train_mask: np.asarray(data.train_mask),
                                                             self.tf_X_binary_mask: np.asarray(data.train_binary_mask),
                                                            self.tf_X_seq_len: np.asarray(data.train_seq_len),
                                                             self.tf_X_sent_for_word: np.asarray(data.train_sentiment_for_word),
                                                             self.tf_y_train: np.asarray(data.train_label),
                                                             self.keep_prob: 0.8})
                
                print('training_accuracy => %.2f, cost value => %.5f for step %d' % \
                (float(correct_prediction_train)/np.sum(np.asarray(data.train_binary_mask)), cost_train, it))
                
                loss_list.append(cost_train)
                accuracy_list.append(float(correct_prediction_train)/np.sum(np.asarray(data.train_binary_mask)))
                
                _, ax1 = plt.subplots()
                ax2 = ax1.twinx()
                ax1.plot(loss_list)
                ax2.plot(accuracy_list, 'r')
                ax1.set_xlabel('epoch')
                ax1.set_ylabel('train loss')
                ax2.set_ylabel('train accuracy')
                ax1.set_title('train accuracy and loss')
                ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
                plt.savefig('accuracy_loss.eps', format='eps', dpi=150)
                plt.close()
                


        self.session.close()


def main():
    batch_size = 128
    sentiment_NB = 3
    Sentiment_NB_Word = 5
    embedding_size = 100
    lstm_nb_inside = 256
    layers = 64
    epochs = 5000
    LEARNING_RATE = 0.01
    WEIGHT_DECAY = 0.001
    label_dict = {
        'aspositive' : 1,
        'asneutral' : 0,
        'asnegative': 2
    }
    Directory = 'C:/karthik/TMP-ABSA/data/'
    Category = 'Restaurants'
    max_seq_len = 19
    NB_Words = 128


    
    File_Change_to_text_File = True
    Sentiment_label = False
    flag_use_sentiment_for_word = True
    flag_train = True

    negative_weight = 2.0
    positive_weight = 1.0
    neutral_weight = 0.0

    session = tf.Session()
    
    data = Data(Category, Directory, label_dict, max_seq_len,
        File_Change_to_text_File, negative_weight, positive_weight, neutral_weight, Sentiment_label)


    if (flag_use_sentiment_for_word):
        embedding_size = embedding_size +  Sentiment_NB_Word

    model = Model(batch_size, max_seq_len, sentiment_NB,  Sentiment_NB_Word, embedding_size,   NB_Words ,
       lstm_nb_inside, layers, epochs, LEARNING_RATE, WEIGHT_DECAY, flag_train, flag_use_sentiment_for_word,session)

    model.modeling()

    if (model.flag_train):
        model.train(data)
    else:
        model.load_model()
        model.evaluate(data, True, model.flag_train)

if __name__ == "__main__":
    main()

