#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 16:29:49 2022
@author: lirubing
"""

import warnings
warnings.filterwarnings("ignore")
import pandas as pd

# --- Read all files
# filename1 = 'wsb_comments_'
# filename2 = '_ticker_present.csv'
# dlst = []
# for i in range(1,13):
#     dlst.append('2020-'+str(i))
# dlst.append('2021-1')
# dlst.append('2021-2')
# files = [(filename1 + i + filename2) for i in dlst]

# take the single file as an example
file = pd.read_csv('wsb_comments_2021-2_ticker_present.csv')
my_data = file[["body", "created_utc"]]



# label the stock mentioned in the comment
# my_data_adj["stock_mentioned"] = 

# take TSLA as an example
Indicators = {'TSLA', 'Tesla', 'Elon Musk'}
c = [True if len(set(my_data.body[i].split()).intersection(Indicators)) != 0 else False for i in range(len(my_data))]
my_data['company'] = c



from utils import *

def Preprocessing(dataset, body):
    txt = body.apply(clean_text)
    txt = txt.apply(rem_sw)
    dataset["adj_txt"] = txt.apply(my_stem)
    return dataset

my_data_adj = Preprocessing(my_data, my_data.body)


# create time stamp
my_data_adj["created_utc"] = my_data_adj["created_utc"].astype(str)
my_data_adj["year"] = my_data_adj["created_utc"].str.slice(0,4)

from datetime import datetime
daylst = my_data_adj["created_utc"].str.slice(5,10)
daylst = [datetime.strptime(i, '%m%d') for i in daylst]
my_data_adj["day"] = daylst

my_data_adj = my_data_adj.drop(columns = ['created_utc'])


# Vectorize
import pickle
x_form = pickle.load(open('vectorizer.pk', 'rb'))
x_txt = x_form.transform(my_data_adj["adj_txt"])


# Scoring with common dictionary
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sent = SentimentIntensityAnalyzer()
my_data_adj["score"] = my_data_adj.adj_txt.apply(sent.polarity_scores)
comp = [float(my_data_adj["score"][i]['compound']) for i in range(len(my_data_adj))]
my_data_adj['score'] = comp

# With finance dictionary --- still working on

# Narrow the score range to +/-0.95 to filter out outliers
