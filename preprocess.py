
import tensorflow as tf
import numpy as np
import os
import re
import nltk
from nltk.corpus import stopwords

from collections import Counter
import codecs
from collections import defaultdict
import xml.etree.ElementTree as ET


def sentiment_dic():
    positiveList  = list()
    negative_list = list()
    reverse_list = list()
    incrementlist = list()
    decrementlist = list()
    sentimentdict = dict()

    fneg = open('C:/karthik/TMP-ABSA/code/dictionary/negative_words.txt', 'r')
    fpos = open('C:/karthik/TMP-ABSA/code/dictionary/positive_words.txt', 'r')
    frev = open('C:/karthik/TMP-ABSA/code/dictionary/reverse_words.txt', 'r')
    fdec = open('C:/karthik/TMP-ABSA/code/dictionary/decremental_words.txt', 'r')
    finc = open('C:/karthik/TMP-ABSA/code/dictionary/incremental_words.txt', 'r')

    for line in fpos:
        if not line.strip() in sentimentdict:
            sentimentdict[line.strip()] = 0
            positiveList.append(line.strip())

    for line in fneg:
        if not line.strip() in sentimentdict:
            sentimentdict[line.strip()] = 1
            negative_list.append(line.strip())

    for line in frev:
        if not line.strip() in sentimentdict:
            sentimentdict[line.strip()] = 2
            reverse_list.append(line.strip())

    for line in finc:
        if not line.strip() in sentimentdict:
            sentimentdict[line.strip()] = 3
            incrementlist.append(line.strip())

    for line in fdec:
        if not line.strip() in sentimentdict:
           sentimentdict[line.strip()] = 4
           decrementlist.append(line.strip())
            
    fneg.close()
    fpos.close()
    frev.close()
    fdec.close()
    finc.close()

    return positiveList,   negative_list,reverse_list, incrementlist, decrementlist,sentimentdict

def File_Change(Category, Directory):
    train_filename = Directory + Category + '_Train.xml'
    test_filename = Directory + Category + '_Test.xml'

    train_text = codecs.open(Directory + Category + '_Train.txt', 'w', 'utf-8')
    test_text = codecs.open(Directory + Category + '_Test.txt', 'w', 'utf-8')

    reviews = ET.parse(train_filename).getroot().findall('Review')
    sentences = []
    for r in reviews:
        sentences += r.find('sentences').getchildren()

    for i in range(len(sentences)):
        try:
            sentence = sentences[i].find('text').text
            new_sentence = sentences[i].find('text').text
            opinions = sentences[i].find('Opinions').findall('Opinion')
            for opinion in opinions:
                start = int(opinion.get('from'))
                end = int(opinion.get('to'))
                polarity = opinion.get('polarity')
                if (end != 0):
                    new_sentence = new_sentence.replace(sentence[start:end],
                                                        sentence[start:end] + '{as' + polarity + '}')
                else:
                    new_sentence = new_sentence + ' food{as' + polarity + '}'
                    
            train_text.write(' '.join(re.sub(r'[.,:;/\-?!\"\n()\\]',' ', new_sentence).lower().split()) + '\n')

        except AttributeError:
            continue

    reviews = ET.parse(test_filename).getroot().findall('Review')
    sentences = []
    for r in reviews:
        sentences += r.find('sentences').getchildren()

    for i in range(len(sentences)):
        try:
            sentence = sentences[i].find('text').text
            new_sentence = sentences[i].find('text').text
            opinions = sentences[i].find('Opinions').findall('Opinion')
            for opinion in opinions:
                start = int(opinion.get('from'))
                end = int(opinion.get('to'))
                polarity = opinion.get('polarity')
                if (end != 0):
                    new_sentence = new_sentence.replace(sentence[start:end],
                                                        sentence[start:end] + '{as' + polarity + '}')
                else:
                    new_sentence = new_sentence + ' food{as' + polarity + '}'
                    
            test_text.write(' '.join(re.sub(r'[.,:;/\-?!\"\n()\\]',' ', new_sentence).lower().split()) + '\n')

        except AttributeError:
            continue
def export_aspect(Category, Directory):
    aspect_list = list()
    
    fa = codecs.open('C:/karthik/TMP-ABSA/code/dictionary/' + Category + '_aspect.txt', 'w', 'utf-8')
    for file in os.listdir(Directory):
        if not (file.endswith('.txt') and Category in file):
            continue
            
        f = codecs.open(Directory + file, 'r', 'utf-8')
        for line in f:
            for word in line.split(' '):
                if '{as' in word:
                    aspect_list.append(word.split('{')[0].strip())
        f.close()
            
    for w in sorted(set(aspect_list)):
        fa.write(w + '\n')
    
    fa.close()
    
    return set(aspect_list)



def data_Label(Category, Directory,Sentiment_label):
    word_dict = dict()
    embedding = list()

    
    Specific_word_senti = defaultdict(list)
    if (Sentiment_label):
        f_se = codecs.open('C:/karthik/TMP-ABSA/code/dictionary/sentiment_specific_words.txt', 'r', 'utf-8')
        
        for line in f_se:
            elements = line.split()
            for i in range(1, len(elements)):
                Specific_word_senti[elements[0].strip()].append(float(elements[i]))
        f_se.close()

    fileV = codecs.open('C:/karthik/TMP-ABSA/data/gensim_word2vec.vec', 'r', 'utf-8')

    idx = 0
    for line in  fileV:
        if len(line) < 50:
            continue
        else:
            component = line.strip().split(' ')
            word_dict[component[0].lower()] = idx
            if (Sentiment_label and component[0].lower() in Specific_word_senti.keys()):
                embedding.append(Specific_word_senti[component[0].lower()])
            else:
                word_vec = list()
                for i in range(1, len(component)):
                    word_vec.append(float(component[i]))
                embedding.append(word_vec)
            idx = idx + 1
    fileV.close()
    word_dict['<padding>'] = idx
    embedding.append([0.] * len(embedding[0]))
    word_dict_rev = {v: k for k, v in word_dict.items()}
    return word_dict, word_dict_rev, embedding



    



def load_data(Category, Directory, label_dict, max_seq_len,
            File_Change_to_text_File, negative_weight, positive_weight, neutral_weight, 
            Sentiment_label):
    train_data = list()
    train_mask = list()
    train_binary_mask = list()
    train_label = list()
    train_seq_len = list()
    train_sentiment_for_word = list()
    test_data = list()
    test_mask = list()
    test_binary_mask = list()
    test_label = list()
    test_seq_len = list()
    test_sentiment_for_word = list()
    count_pos = 0
    count_neg = 0
    count_neu = 0

    if (File_Change_to_text_File):
        File_Change(Category, Directory)


    positiveList,   negative_list,reverse_list, incrementlist, decrementlist,sentimentdict = sentiment_dic()
    aspect_list = export_aspect(Category, Directory)
    word_dict, word_dict_rev, embedding = data_Label(Category, Directory, Sentiment_label)
    # load data, mask, label
    for file in os.listdir(Directory):
        if not (file.endswith('.txt') and Category in file):
            continue

        f_processed = codecs.open(Directory + file, 'r', 'utf-8')
        for line in f_processed:
            data_tmp = list()
            mask_tmp = list()
            binary_mask_tmp = list()
            label_tmp = list()
            sentiment_for_word_tmp = list()
            count_len = 0

            words = line.strip().split(' ')
            for word in words:
                if (word in stopwords.words('english')):
                    continue
                word_clean = word.replace('{aspositive}', '').replace('{asnegative}', '').replace('{asneutral}', '')

                if (word_clean in word_dict.keys() and count_len < max_seq_len):
                    if (word_clean in positiveList):
                        sentiment_for_word_tmp.append(1)
                    elif (word_clean in   negative_list):
                        sentiment_for_word_tmp.append(2)
                    elif (word_clean in reverse_list):
                        sentiment_for_word_tmp.append(0)
                    elif (word_clean in incrementlist):
                        sentiment_for_word_tmp.append(0)
                    elif (word_clean in decrementlist):
                        sentiment_for_word_tmp.append(0)
                    else:
                        sentiment_for_word_tmp.append(0)

                    if ('aspositive' in word):
                        mask_tmp.append(positive_weight)
                        binary_mask_tmp.append(1.0)
                        label_tmp.append(label_dict['aspositive'])
                        count_pos = count_pos + 1
                    elif ('asneutral' in word):
                        mask_tmp.append(neutral_weight)
                        binary_mask_tmp.append(1.0)
                        label_tmp.append(label_dict['asneutral'])
                        count_neu = count_neu + 1
                    elif ('asnegative' in word):
                        mask_tmp.append(negative_weight)
                        binary_mask_tmp.append(1.0)
                        label_tmp.append(label_dict['asnegative'])
                        count_neg = count_neg + 1
                    else:
                        mask_tmp.append(0.)
                        binary_mask_tmp.append(0.)
                        label_tmp.append(0)
                    count_len = count_len + 1

                    data_tmp.append(word_dict[word_clean])
                elif '{as' in word and file != Category + '_Train_Final.txt':
                    print(word)

            if file == Category + '_Train.txt':
                train_seq_len.append(count_len)
            else:
                test_seq_len.append(count_len)

            for _ in range(max_seq_len - count_len):
                data_tmp.append(word_dict['<padding>'])
                mask_tmp.append(0.)
                binary_mask_tmp.append(0.)
                label_tmp.append(0)
                sentiment_for_word_tmp.append(0)

            if file == Category + '_Train.txt':
                train_data.append(data_tmp)
                train_mask.append(mask_tmp)
                train_binary_mask.append(binary_mask_tmp)
                train_label.append(label_tmp)
                train_sentiment_for_word.append(sentiment_for_word_tmp)
            else:
                test_data.append(data_tmp)
                test_mask.append(mask_tmp)
                test_binary_mask.append(binary_mask_tmp)
                test_label.append(label_tmp)
                test_sentiment_for_word.append(sentiment_for_word_tmp)
        f_processed.close()

    print('positive reviews: %d' %count_pos)
    print('neutral reviews: %d' %count_neu)
    print('negative reviews: %d' %count_neg)
    print('length of train data is %d' %(len(train_data)))
    print('length of test data is %d' %(len(test_data)))
    data_sample = ''
    for id in train_data[10]:
        data_sample = data_sample + ' ' + word_dict_rev[id]

    print('%s' %data_sample)
    print(train_data[10])
    print(train_mask[10])
    print(train_label[10])
    print(train_sentiment_for_word[10])
    print('length of word dictionary is %d' %(len(word_dict)))
    print('length of embedding is %d' %(len(embedding)))
    print('length of Aspect list is %d' %(len(aspect_list)))
    print('maximum sequence length is %d' %(np.max(test_seq_len)))

    return train_data, train_mask, train_binary_mask, train_label, train_seq_len, train_sentiment_for_word, \
    test_data, test_mask, test_binary_mask, test_label, test_seq_len, test_sentiment_for_word, \
    word_dict, word_dict_rev, embedding, aspect_list


def main():
    max_seq_len = 20
    negative_weight = 2.0
    positive_weight = 1.0
    neutral_weight = 0.0

    label_dict = {
        'aspositive' : 1,
        'asneutral' : 0,
        'asnegative': 2
    }

    Directory = 'C:/karthik/TMP-ABSA/data/'
    Category = 'Restaurants'
  
    File_Change_to_text_File = True
    Sentiment_label = False

    train_data, train_mask, train_binary_mask, train_label, train_seq_len, train_sentiment_for_word, \
    test_data, test_mask, test_binary_mask, test_label, test_seq_len, test_sentiment_for_word, \
    word_dict, word_dict_rev, embedding, aspect_list = load_data(
        Category,
        Directory,
        label_dict,
        max_seq_len,
        File_Change_to_text_File,
        negative_weight,
        positive_weight,
        neutral_weight,
        Sentiment_label
    )

if __name__ == '__main__':
    main()