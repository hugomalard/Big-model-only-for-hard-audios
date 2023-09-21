import sys
import os
import numpy as np
import pandas as pd
import random
import re

def werFile2np(werPath):
    werl = []
    idsl = []
    with open(werPath) as file:
        for i,item in enumerate(file):
            match, ids,wer = get_wer(item)
            if match and i > 3:
                werl.append(wer)
                idsl.append(ids)

    wer = np.array(werl)
    ids = np.array(idsl)
    return wer,ids

def get_wer(line):
    id = 0
    wer=100
    match = re.search(r'%WER (\d+\.\d+)', line)
    if match:
        id = line.split()[0][:-1]
        wer = float(match.group(1))
    return match, id,wer

def add_choice(df): 
    choice = np.array(df['werTiny'] > df['werSmall'])
    df['choice'] = choice.astype(int).tolist()
    return df

df_test = pd.read_csv('BMOHA/results/save/test.csv')
df_dev = pd.read_csv('BMOHA/results/save/dev.csv')
df_train = pd.read_csv('BMOHA/results/save/train.csv')

wer_tiny_train,id_tiny_train = werFile2np('BMOHA/results/inferenceTiny/train_wer.txt')
wer_small_train,id_small_train = werFile2np('BMOHA/results/inferenceSmall/train_wer.txt')

wer_tiny_dev,id_tiny_dev = werFile2np('BMOHA/results/inferencesTiny/dev_wer.txt') 
wer_small_dev,id_small_dev = werFile2np('BMOHA/results/inferenceSmall/dev_wer.txt') 

wer_tiny_test,id_tiny_test = werFile2np('BMOHA/results/inferenceTiny/test_wer.txt') 
wer_small_test,id_small_test = werFile2np('BMOHA/results/inferenceSmall/test_wer.txt') 


train_wer = pd.merge(pd.DataFrame({'werTiny': wer_tiny_train, 'ID': id_tiny_train}),pd.DataFrame({'werSmall': wer_small_train, 'ID': id_small_train}),on='ID')
dev_wer = pd.merge(pd.DataFrame({'werTiny': wer_tiny_dev, 'ID': id_tiny_dev}),pd.DataFrame({'werSmall': wer_small_dev, 'ID': id_small_dev}),on='ID')
test_wer = pd.merge(pd.DataFrame({'werTiny': wer_tiny_test, 'ID': id_tiny_test}),pd.DataFrame({'werSmall': wer_small_test, 'ID': id_small_test}),on='ID')

train_set = pd.merge(df_train,train_wer,on='ID')
dev_set = pd.merge(df_dev,dev_wer,on='ID')
test_set = pd.merge(df_test,test_wer,on='ID')

train_set = add_choice(train_set)
dev_set = add_choice(dev_set)
test_set = add_choice(test_set)

train_set.to_csv('BMOHA/results/save/train-WER.csv')
dev_set.to_csv('BMOHA/results/save/dev-WER.csv')
test_set.to_csv('BMOHA/results/save/test-WER.csv')
