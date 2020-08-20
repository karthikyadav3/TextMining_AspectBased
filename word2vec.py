
"""
Created on Sun Sep 15 10:21:59 2019

@author: Sai Karthik Yadav
"""

import gensim
import glob
import numpy as np

files=glob.glob("C:/karthik/aspect-based-sentiment-analysis-master/data/vec/*.txt")
with open('C:/karthik/aspect-based-sentiment-analysis-master/data/vec/Laptops_Train_Final.txt','r',encoding='utf-8') as f:
    #print(f.read())

     review_list= []
for file in files:
    try:
        with open(file,'r',encoding='utf-8') as f:
            review_list.append(f.read())
    except:
        pass
            

clean_list= []
for text in review_list:
    clean_list.append(gensim.utils.simple_preprocess(text))

model = gensim.models.Word2Vec(
        clean_list,
        size=150,
        window=10,
        min_count=5,
        workers=10)

model.train(clean_list, total_examples=len(clean_list), epochs=10)

words = list(model.wv.vocab)
vec = []
for word in words:
    vec.append(model[word].tolist())
data = np.array(vec)
data
f= open("C:/karthik/aspect-based-sentiment-analysis-master/data/vecgensim_word2vec.vec","w+")
with open('C:/karthik/aspect-based-sentiment-analysis-master/data/vecgensim_word2vec.vec', 'w') as f:
    for item in data:
        f.write("%s\n" % item)