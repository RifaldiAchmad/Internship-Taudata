# -*- coding: utf-8 -*-
"""
Copyright Taufik Sutanto - taudata Analytics http://taudata-analytics.com
"""

import warnings; warnings.simplefilter('ignore')
import tauData as tau, tauModels as tauM
import torch, gc, os, pickle
import pandas as pd, numpy as np  
from sklearn.model_selection import train_test_split
from tqdm import tqdm; tqdm.pandas() 
from transformers import BertTokenizer
from transformers import BertForSequenceClassification#, AdamW, BertConfig
from transformers import logging; logging.set_verbosity_error()
import torch.nn.functional as F
np.random.seed(787)

if __name__ == '__main__':
    # Loading and-or preprocessing the data
    Resume = False
    train_size = 0.90
    test_size = 0.10
    model_max_length = 256
    epochs = 3
    batch_size = 32
    fData = 'data/judul-berita-kategori.zip'
    fDataCleaned = 'data/judul-berita-kategori-cleaned.csv'
    fLabel2Num = 'data/label2num.pckl' #Dictionary Label to Numeric, to ensure concistency
    output_dir = 'models/'
    fStats = 'data/df_stats.csv'
    print("Loading Data ... ")
    try:
        df = pd.read_csv(fDataCleaned.replace(".csv", ".zip"), compression='zip')
        f = open(fLabel2Num, 'rb')
        label2num, num2label = pickle.load(f); f.close()
        #df['bidang'] = df.topik.map(num2label)
        #df.to_csv(fDataCleaned, index=False, encoding='utf8')
    except:
        df = pd.read_csv(fData, compression='zip')
        print(df.shape)

        topics = set(df.topik)
        label2num = {t:i for t,i in zip(topics, range(len(topics)))}
        num2label = {i:t for t,i in zip(topics, range(len(topics)))}
        f = open(fLabel2Num, 'wb')
        pickle.dump((label2num, num2label), f); f.close() 
        
        print(label2num)
        df.topik = df.topik.map(label2num)
        df.judul = df.judul.progress_apply(lambda x: tau.cleanText(x, maxWords=model_max_length)) # .progress_map(tauS.cleanText)
        df.to_csv(fDataCleaned, index=False, encoding='utf8')
        tau.compress(fDataCleaned, ext="csv", level=9, delete=True) #file_, ext="csv", level=9, delete=True
        
    num_labels = len(label2num)
    y = df.pop('topik').values
    X = df
    train_sentence, test_sentences, train_labels, test_labels = train_test_split(X.index,y,test_size=test_size)
    train_sentence = [sent[0] for sent in X.iloc[train_sentence].values] # return dataframe train
    test_sentences = [sent[0] for sent in X.iloc[test_sentences].values]
    
    # Checking GPU, loading device, and cleaning CPU-GPU memory
    print("Initializing CPU and-or GPU ... ")
    gc.collect()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
    torch.cuda.empty_cache()
    device = tauM.gpuTest()
    
    if Resume:
        tokenizer, model = tauM.loadModel(device, BertForSequenceClassification, BertTokenizer, 
                                          output_dir=output_dir)
    else: # Train the model
        tokenizer, model = tauM.trainModel(train_sentence, train_labels, device, num_labels, 
                                           train_size=train_size, epochs=epochs, 
                                           model_max_length=model_max_length, batch_size=batch_size, 
                                           output_dir=output_dir)
    
    test_judul = "pendidikan sangat bergantung pada kurikulum"
    print(test_judul)
    test_judul = tau.cleanText(test_judul, maxWords=model_max_length)
    encoding = tokenizer.encode_plus(
                        test_judul,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = model_max_length,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                        truncation=True, ).to(device)
    seqOutput = model(encoding["input_ids"], encoding["attention_mask"])
    probabilities = seqOutput.logits.detach().cpu().numpy()
    probabilities = F.softmax(torch.tensor(probabilities[0]), dim=0)
    prediction = num2label[int(np.argmax(probabilities, axis=0).flatten()[0].cpu().numpy())]
    print("kategori Berita: ", prediction)
    
    
    """
    Kalau code sudah jalan tinggal di Loop utk predict semua data yang ada.
    """