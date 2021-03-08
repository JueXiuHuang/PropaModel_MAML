#!/usr/bin/env python
# coding: utf-8

# # PropaModel_MAML
# ## description
# 因為標記資料較少，如果使用傳統的類神經網路分類方法效果不佳，因此試圖使用few-shot learning (meta learning)來改善成效。
# ## related work
# 主要參考以下[程式碼](https://colab.research.google.com/github/mari-linhares/tensorflow-maml/blob/master/maml.ipynb#scrollTo=9_OOYo7NB2aI)
# [MAML paper](https://arxiv.org/pdf/1703.03400.pdf)
# [參考鏈結1](https://zhuanlan.zhihu.com/p/57864886)
# ## few-shot learning with MAML
# ![image.png](attachment:image.png)
# ## current dataset label distribution
# ![image-2.png](attachment:image-2.png)

# In[1]:


from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import shuffle, class_weight
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from transformers import *
from collections import Counter
from official import nlp
from random import seed, randint
import official.nlp.optimization
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import seaborn as sns
import math
import random
import re
import sys
import time


# In[2]:


def load_dataset():
    dfUDN = pd.read_csv('originalDataset/propa/UDN-bootstrap-checked.tsv', sep='\t', names=['宣傳手法', '標記詞', '句子'])
    dfLT = pd.read_csv('originalDataset/propa/LT-bootstrap-checked.tsv', sep='\t', names=['宣傳手法', '標記詞', '句子'])
    return dfUDN, dfLT

def generate_sentence(X):
    # 去除非中文字元(包含標點符號)
    for i in range(len(X)):
        X[i] = re.sub(r'[^\u4e00-\u9fa5]', '', X[i])
    return X

def encode_data(X, tokenizer):
    input_ids, token_type_ids, attention_mask = [], [], []
    for i in range(len(X)):
        inputs = tokenizer.encode_plus(X[i],add_special_tokens=True, max_length=64, pad_to_max_length=True,
            return_attention_mask=True, return_token_type_ids=True, truncation=True)
        input_ids.append(inputs['input_ids'])
        token_type_ids.append(inputs['token_type_ids'])
        attention_mask.append(inputs['attention_mask']) 
    return np.asarray(input_ids, dtype='int32'), np.asarray(attention_mask, dtype='int32'), np.asarray(token_type_ids, dtype='int32')

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()
    return


# In[3]:


def build_embedding_generator(model_base):
    model = TFBertModel.from_pretrained(model_base, return_dict=True)
    model.layers[0].trainable = False
    input1 = tf.keras.Input(shape=(64,), dtype=tf.int32, name='input_ids')
    input2 = tf.keras.Input(shape=(64,), dtype=tf.int32, name='attention_mask')
    input3 = tf.keras.Input(shape=(64,), dtype=tf.int32, name='token_type_ids')
    outDict = model([input1, input2, input3])
    #print(outDict.keys())
    # >>> odict_keys(['last_hidden_state', 'pooler_output'])
    #print(outDict['last_hidden_state'].shape)
    # >>> (None, 64, 768)
    #print(outDict['pooler_output'].shape)
    # >>> (None, 768)
    #outs = outDict['last_hidden_state']
    outs = outDict['pooler_output']
    model = tf.keras.Model(inputs=[input1, input2, input3], outputs=outs)
    #model.summary()
    return model


# In[4]:


def sample_tasks(x, y, mode, not_seen, c1, c2):
    '''
    Args:
        test: the label only shown in testing
    '''
    all_class = [['LL', 'LLL'],          # total 511
                   ['FW'],               # total 29
                   ['BW', 'DS', 'BWF'],  # total 15
                   ['PS'],               # total 48
                   ['NCL'],              # total 87
                   ['ATA'],              # total 22
                   ['ATF'],              # total 6
                   ['DT'],               # total 15
                   ['TTC'],              # total 6
                   ['HA'],               # total 9
                   ['EXAGGERATION']]     # total 3
    
    # k shots
    shots = 10
    class_i = [[], [], [], [], [], [], []]
    
    # 把 embedding依照label分成七類 0, 1, 2, 3, 4, 5, others
    for i in range(len(label)):
        if any(c in label[i] for c in all_class[0]):
            class_i[0].append(x[i])
        elif any(c in label[i] for c in all_class[1]):
            class_i[1].append(x[i])
        elif any(c in label[i] for c in all_class[2]):
            class_i[2].append(x[i])
        elif any(c in label[i] for c in all_class[3]):
            class_i[3].append(x[i])
        elif any(c in label[i] for c in all_class[4]):
            class_i[4].append(x[i])
        elif any(c in label[i] for c in all_class[5]):
            class_i[5].append(x[i])
        else:
            class_i[6].append(x[i])
    
    # 把 other的embedding放在class_i 的倒數第二個
    class_i[5], class_i[6] = class_i[6], class_i[5]
    # 把 test的embedding放在class_i 的最後一個
    class_i[not_seen], class_i[6] = class_i[6], class_i[not_seen]
    
    r = []
    if mode == 'train':
        # 產生一組K shot task
        inner_x = []
        inner_y = []
        inner_x = inner_x + random.sample(class_i[c1], shots)
        inner_y = inner_y + [1]*shots
        inner_x = inner_x + random.sample(class_i[c2], shots)
        inner_y = inner_y + [0]*shots
        inner_x, inner_y = shuffle(np.array(inner_x), np.array(inner_y))
        r.append((np.array(inner_x), np.array(inner_y)))
        
        # 產生一組 meta update task
        outer_x = []
        outer_y = []
        
        outer_x = outer_x + random.sample(class_i[c1], 2)
        outer_y = outer_y + [1]*2
        outer_x = outer_x + random.sample(class_i[c2], 2)
        outer_y = outer_y + [0]*2
        outer_x, outer_y = shuffle(np.array(outer_x), np.array(outer_y))
        r.append((np.array(outer_x), np.array(outer_y)))
        
    elif mode == 'finetune':
        # 產生finetune task
        finetune_x = []
        finetune_y = []
        test_x = []
        test_y = []
        for t in range(7):
            P_fine, P_test = train_test_split(class_i[t], test_size=0.2)
            #finetune_x = finetune_x + random.sample(P_fine, int(len(P_fine)/2))
            finetune_x = finetune_x + random.sample(P_fine, 5)
            test_x = test_x + random.sample(P_test, len(P_test))
            if t == 6:
                #finetune_y = finetune_y + [1]*int(len(P_fine)/2)
                finetune_y = finetune_y + [1]*5
                test_y = test_y + [1]*len(P_test)
            else:
                #finetune_y = finetune_y + [0]*int(len(P_fine)/2)
                finetune_y = finetune_y + [0]*5
                test_y = test_y + [0]*len(P_test)
        finetune_x, finetune_y = shuffle(np.array(finetune_x), np.array(finetune_y))
        test_x, test_y = shuffle(np.array(test_x), np.array(test_y))
        r.append((finetune_x, finetune_y))
        r.append((test_x, test_y))
    
    return r


# In[5]:


# create a binary classifier
def create_model():
    inp = keras.Input(shape=(768,))
    h1 = keras.layers.Dense(128, activation='relu')(inp)
    h2 = keras.layers.Dense(64, activation='relu')(h1)
    out = keras.layers.Dense(1, activation='sigmoid')(h2)
    
    m = keras.Model(inputs=inp, outputs=out)
    
    return m


# In[6]:


def copy_model(model):
    #Copy model weights to a new model.
    copied_model = create_model()
    
    #copied_model.set_weights(model.get_weights())
    for a, b in zip(copied_model.variables, model.variables):
        a.assign(b)
    return copied_model

def train_maml(model, epochs, dataset, lr_alpha, lr_beta, batch_size, target):
    '''
    Train using the MAML setup.
    
    The comments in this function that start with:
        
        Step X:
        
    Refer to a step described in the Algorithm 1 of the paper.
    '''
    # Instantiate an optimizer and loss function
    optimizer = keras.optimizers.Adam(learning_rate=0.003)
    loss_fn = keras.losses.BinaryCrossentropy()
    
    x, y = dataset
    
    train_loss_results = []
    
    # Step 2: instead of checking for convergence, we train for a number of epochs
    for _ in range(epochs):
        meta_tests = []
        total_grads = []
        
        # Step 3: Sample batch of tasks
        batch_of_tasks = []
        for t in range(7):
            t1 = randint(0,5)
            t2 = t1
            while t1 == t2:
                t2 = randint(0,5)
            batch_of_tasks.append((t1, t2)) 

        # Step 4
        for t in range(len(batch_of_tasks)):
            # Step 5 & 8
            t1, t2 = batch_of_tasks[t]
            training_data = sample_tasks(x, y, 'train', target, t1, t2)
            inner_x, inner_y = training_data[0]
            outer_x, outer_y = training_data[1]

            # Step 6
            model_copy = copy_model(model)
            for inn_epoch in range(50):
                with tf.GradientTape() as train_tape:
                    logits = model_copy(inner_x, training=True)
                    loss_value = loss_fn(inner_y, logits)
                grads = train_tape.gradient(loss_value, model_copy.trainable_variables)
                optimizer.apply_gradients(zip(grads, model_copy.trainable_variables))

            print('=====================================')
            # Step 10
            with tf.GradientTape() as test_tape:
                logits = model_copy(outer_x, training=True)
                loss_value = loss_fn(outer_y, logits)
                if _ > 70 and _%10 == 0:
                    pass
                train_loss_results.append(float(loss_value))
            grads = test_tape.gradient(loss_value, model_copy.trainable_variables)
            if len(total_grads) != 0:
                for g in range(len(total_grads)):
                    total_grads[g] += grads[g]
            else:
                total_grads = grads

        optimizer.apply_gradients(zip(total_grads, model.trainable_variables))
        if _%10 == 0:
            print("epoch: {:>3d}, loss: {:6.4f}".format(_,train_loss_results[_]))
            
    title = 'Meta train loss'
    plt.figure(figsize=(12, 8))
    plt.plot(train_loss_results, label='loss_value')
    plt.title(title, fontsize=15)
    plt.xlabel('epochs', fontsize=15)
    plt.ylabel('loss value', fontsize=15)
    plt.show()
        
    return model


# In[7]:


def finetune_and_eval_maml(model, epochs, dataset, lr, target):
    x, y = dataset
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    loss_fn = keras.losses.BinaryCrossentropy()
    accuracy = keras.metrics.BinaryAccuracy()
    
    finetune_loss = []
    finetune_acc = []
    
    data = sample_tasks(x, y, 'finetune', target, 0, 1)
    test_x, test_y = data[1]
    
    fx, fy = data[0]
    
    for _ in range(epochs):
        fx, fy = shuffle(fx, fy)
        with tf.GradientTape() as tape:
            logits = model(fx, training=True)
            loss_value = loss_fn(fy, logits)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        accuracy.update_state(fy, logits)
        print("Epoch: {:>3d}, Loss: {:6.4f}, accuracy: {:6.4f}".format(_, float(loss_value), accuracy.result()))
        finetune_loss.append(float(loss_value))
        finetune_acc.append(accuracy.result())

        #step = step + 1

    title = 'Fintuning Metrics'
    plt.figure(figsize=(12, 8))
    plt.plot(finetune_loss, color='g',marker='o', label='loss_value')
    plt.plot(finetune_acc, color='b',marker='v', label='accuracy')
    plt.title(title, fontsize=15)
    plt.xlabel('epochs', fontsize=15)
    plt.ylabel('value of loss and accuracy', fontsize=15)
    plt.legend(fontsize='xx-large')
    plt.show()
    
    logits_pre = model(test_x, training=False)
    pred = tf.round(logits_pre)
    print(classification_report(test_y, pred))
    
    return model


# In[8]:


# Main function
if __name__ == '__main__':
    seed(1)
    dfUDN, dfLT = load_dataset()

    df = pd.concat([dfUDN, dfLT], ignore_index=True)
    label = df['宣傳手法'].to_numpy()
    sentences = df['句子'].to_numpy()
    X = generate_sentence(sentences)

    model_base = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_base)
    model_emb = build_embedding_generator(model_base)


# In[9]:


# generate the word embedding with TFbert model
input_ids, token_type_ids, attention_mask = encode_data(X, tokenizer)
emb = model_emb.predict([input_ids, token_type_ids, attention_mask])
#emb = emb.reshape(emb.shape[0], -1)


# In[10]:


emb.shape


# In[11]:


#model_maml = ClassifyModel()
model_maml = create_model()
model_maml.summary()


# In[12]:


model_maml = train_maml(model_maml, 100, (emb, label), 0.001, 0.001, 1, target=0)


# In[13]:


finetuned_maml = finetune_and_eval_maml(model_maml, 5, (emb, label), 0.003, target=0)


# In[ ]:




