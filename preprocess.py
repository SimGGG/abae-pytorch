import nltk
import os
import re
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import codecs
import json
from tqdm import tqdm
from collections import defaultdict
import argparse

if os.path.exists('../../nltk_data'):
    pass
else:    
    nltk.download('stopwords')
    nltk.download('wordnet')

data = pd.read_csv('datasets/listings.csv', encoding='utf-8').dropna(subset=['description'])

room_type = data['room_type']
sentences = data['description']
regex_text = re.compile('[^a-zA-Z]+')

def parseSentence(line):
    lmtzr = WordNetLemmatizer()
    stop = stopwords.words('english')
    stop.extend(stopwords.words('spanish'))
    stop.extend(['br'])
    text_token = CountVectorizer().build_tokenizer()(line.lower())
    text_rmstop = [i for i in text_token if i not in stop]
    text_stem = [lmtzr.lemmatize(w) for w in text_rmstop]
    return text_stem


room_sent_dict = defaultdict(list)

with open('preprocessed_data/listings.txt', 'w') as txt_out, open('preprocessed_data/listings.pickle', 'wb') as pkl_out:

    for room, line in tqdm(zip(room_type, sentences), total=len(data)):
        line = regex_text.sub(' ', line)
        tokens = ' '.join(parseSentence(line))
        if len(tokens) > 0:
            txt_out.write(tokens + '\n')

            room_sent_dict['room_type'].append(room)
            room_sent_dict['sentences'].append(line)

    pickle.dump(room_sent_dict, pkl_out)
    print('Preprocessing Done!')




