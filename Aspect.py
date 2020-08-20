


import codecs
import math
import numpy as np
import preprocess
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import confusion_matrix
import seaborn as sns


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


    def evaluate(self, data, flag_write_to_file, flag_train):
        if (not flag_train):
            self.load_model()
       
        #Code : Lijin
        original_labels = list()
        pred_labels = list()
       
        original_labels_pos = list()
        original_labels_neu = list()
        original_labels_neg = list()
        pred_labels_pos = list()
        pred_labels_neu = list()
        pred_labels_neg = list()
        #Code : Lijin
       
        correct_prediction_test, prediction_test = self.session.run([self.correct_prediction, self.prediction],
                                              feed_dict={self.tf_X_train: np.asarray(data.testX),
                                                         self.tf_X_binary_mask: np.asarray(data.test_binary_mask),
                                                         self.tf_X_seq_len: np.asarray(data.test_seq_len),
                                                         self.tf_X_sent_for_word: np.asarray(data.test_sentiment_for_word),
                                                         self.tf_y_train: np.asarray(data.test_label),
                                                         self.keep_prob: 1.0})

        print('test accuracy => %.2f' %(float(correct_prediction_test)/np.sum(data.test_binary_mask)))

        if float(correct_prediction_test)/np.sum(data.test_binary_mask) > 0.809:
            self.save_model()

        if (flag_write_to_file):
            f_result = codecs.open('C:/karthik/TMP-ABSA/code/Restaurants/result/result.txt', 'w', 'utf-8')
            f_result.write('#---------------------------------------------------------------------------------------------------------#\n')
           
            f_result.write('#\t test accuracy %.2f\n' %(float(correct_prediction_test)*100/np.sum(np.asarray(data.test_binary_mask) > 0.)))
            f_result.write('#\t 1:positive, 0:neutral, 2:negative\n')
            f_result.write('#---------------------------------------------------------------------------------------------------------#\n')

            for i in range(len(data.test_data)):
                data_sample = ''
                for j in range(len(data.test_data[i])):
                    if data.word_dict_rev[data.test_data[i][j]] == '<unk>':
                        continue
                    elif data.test_binary_mask[i][j] > 0.:
                        data_sample = data_sample + data.word_dict_rev[data.test_data[i][j]] + '(label ' + str(data.test_label[i][j]) + \
                         '|predict ' + str(prediction_test[i][j]) + ') '
                       
                        #Trying to find the values for overall confusion matrix
                        original_labels.append(data.test_label[i][j])
                        pred_labels.append(prediction_test[i][j])
                        #Trying to find the values for overall confusion matrix
                       
                        ##Trying to find the values for confusion matrix individually
                        if(data.test_label[i][j] == 1):
                            original_labels_pos.append('Correct')
                            if(prediction_test[i][j] == 1):
                               pred_labels_pos.append('Correct')
                            else:
                               pred_labels_pos.append('Incorrect')
                        else:
                            original_labels_pos.append('Incorrect')
                            if(prediction_test[i][j] == 1):
                               pred_labels_pos.append('Correct')
                            else:
                               pred_labels_pos.append('Incorrect')
                               
                        if(data.test_label[i][j] == 0):
                            original_labels_neu.append('Correct')
                            if(prediction_test[i][j] == 0):
                               pred_labels_neu.append('Correct')
                            else:
                               pred_labels_neu.append('Incorrect')
                        else:
                            original_labels_neu.append('Incorrect')
                            if(prediction_test[i][j] == 0):
                               pred_labels_neu.append('Correct')
                            else:
                               pred_labels_neu.append('Incorrect')
                               
                        if(data.test_label[i][j] == 2):
                            original_labels_neg.append('Correct')
                            if(prediction_test[i][j] == 2):
                               pred_labels_neg.append('Correct')
                            else:
                               pred_labels_neg.append('Incorrect')
                        else:
                            original_labels_neg.append('Incorrect')
                            if(prediction_test[i][j] == 2):
                               pred_labels_neg.append('Correct')
                            else:
                               pred_labels_neg.append('Incorrect')                            
                        ##Trying to find the values for confusion matrix individually
                       
                    else:
                        data_sample = data_sample + data.word_dict_rev[data.test_data[i][j]] + ' '
                f_result.write('%s\n' %data_sample.replace('<padding>', '').strip())
           
            f_result.close()
           
            #Plot overall confusion matrix
            labels = [0,1,2]
            cm = confusion_matrix(original_labels, pred_labels, labels)
            print(cm)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
            cax = ax.matshow(cm)
            plt.title('Confusion matrix for reviews')
            fig.colorbar(cax)
            ax.set_xticklabels([''] + labels)
            ax.set_yticklabels([''] + labels)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            #plt.show()
            plt.savefig('Confusion_matrix_Overall.png', format='png', dpi=150)
            #Plot overall confusion matrix
           
            #Plot overall confusion matrix
            labels = ['Correct','Incorrect']
            cm = confusion_matrix(original_labels_pos, pred_labels_pos, labels)
            print(cm)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
            cax = ax.matshow(cm)
            plt.title('Confusion matrix for positive reviews')
            fig.colorbar(cax)
            ax.set_xticklabels([''] + labels)
            ax.set_yticklabels([''] + labels)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            #plt.show()
            plt.savefig('Confusion_matrix_POSITIVE.png', format='png', dpi=150)
           
            labels = ['Correct','Incorrect']
            cm = confusion_matrix(original_labels_neu, pred_labels_neu, labels)
            print(cm)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
            cax = ax.matshow(cm)
            plt.title('Confusion matrix for neutral reviews')
            fig.colorbar(cax)
            ax.set_xticklabels([''] + labels)
            ax.set_yticklabels([''] + labels)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            #plt.show()
            plt.savefig('Confusion_matrix_NEUTRAL.png', format='png', dpi=150)
           
            labels = ['Correct','Incorrect']
            cm = confusion_matrix(original_labels_neg, pred_labels_neg, labels)
            print(cm)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
            cax = ax.matshow(cm)
            plt.title('Confusion matrix for negative reviews')
            fig.colorbar(cax)
            ax.set_xticklabels([''] + labels)
            ax.set_yticklabels([''] + labels)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            #plt.show()
            plt.savefig('Confusion_matrix_NEGATIVE.png', format='png', dpi=150)
            #Plot overall confusion matrix

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
