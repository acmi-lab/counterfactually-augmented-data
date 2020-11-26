#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import random
np.random.seed(123)
random.seed(123)
#import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
torch.manual_seed(123)
import torch.optim as optim
from torchtext.data import Field
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from simple_lstm import LSTM
from simple_lstm import save_metrics, load_metrics, save_checkpoint, load_checkpoint

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# # There are no repeated batch_id values --> we can use them as IDs

# In[3]:


path = 'sentiment/combined/paired/{}_paired.tsv'
train_path = path.format('train')
val_path = path.format('dev')
test_path = path.format('test')

train_df = pd.read_table(train_path)
val_df = pd.read_table(val_path)
test_df = pd.read_table(test_path)

all_ids = np.concatenate((train_df['batch_id'].unique(), val_df['batch_id'].unique(), test_df['batch_id'].unique()))
print(f'unique IDs across all 3 sets: {len(np.unique(all_ids))}')
sum_of_all = len(train_df['batch_id'].unique()) + len(val_df['batch_id'].unique()) + len(test_df['batch_id'].unique())
print(f"sum of unique IDs in each: {sum_of_all}")


# # Preprocessing

# In[14]:


# load data
path = 'sentiment/combined/paired/{}_paired.tsv'
train_path = path.format('train')
val_path = path.format('dev')
test_path = path.format('test')

train_df = pd.read_table(train_path)
val_df = pd.read_table(val_path)
test_df = pd.read_table(test_path)

# build text/label fields on factual and counterfactual data
all_train_texts = train_df["Text"].tolist()
label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
text_field = Field(tokenize='spacy', lower=True, include_lengths=True, batch_first=True)
text_field.build_vocab(all_train_texts, min_freq=3)

# setup tokenizer
tokenizer = Tokenizer(num_words=len(text_field.vocab), oov_token=True)
tokenizer.fit_on_texts(all_train_texts)

# split into factual and counterfactual
# we'll do this now for simplicity, at the cost of some copy-pasting below
train_IDs = train_df['batch_id'].values
val_IDs = val_df['batch_id'].values
test_IDs = test_df['batch_id'].values

indices = list(range(len(train_df)))
factual_indices, counterfactual_indices = train_test_split(indices, test_size=0.5, stratify=train_IDs)
#print(factual_indices)
temp = factual_indices + counterfactual_indices
temp2 = counterfactual_indices + factual_indices
factual_indices = temp
counterfactual_indices = temp2

cf_train_df = train_df.iloc[counterfactual_indices]
train_df = train_df.iloc[factual_indices]

"""
plt.hist(train_df['Sentiment'])
plt.title('Train Factuals')
plt.figure()
plt.hist(cf_train_df['Sentiment'])
plt.title('Train Counterfactuals')
plt.show()
"""
indices = list(range(len(val_df)))
factual_indices, counterfactual_indices = train_test_split(indices, test_size=0.5, stratify=val_IDs)
cf_val_df = val_df.iloc[counterfactual_indices]
val_df = val_df.iloc[factual_indices]

indices = list(range(len(test_df)))
factual_indices, counterfactual_indices = train_test_split(indices, test_size=0.5, stratify=test_IDs)
cf_test_df = test_df.iloc[counterfactual_indices]
test_df = test_df.iloc[factual_indices]


# load text, labels, and IDs
train_IDs = train_df['batch_id'].tolist()
val_IDs = val_df['batch_id'].tolist()
test_IDs = test_df['batch_id'].tolist()

cf_train_IDs = cf_train_df['batch_id'].tolist()
cf_val_IDs = cf_val_df['batch_id'].tolist()
cf_test_IDs = cf_test_df['batch_id'].tolist()

train_texts = train_df['Text'].tolist()
val_texts = val_df['Text'].tolist()
test_texts = test_df['Text'].tolist()

cf_train_texts = cf_train_df['Text'].tolist()
cf_val_texts = cf_val_df['Text'].tolist()
cf_test_texts = cf_test_df['Text'].tolist()

train_labels = (train_df['Sentiment'] == 'Positive').tolist()
val_labels = (val_df['Sentiment'] == 'Positive').tolist()
test_labels = (test_df['Sentiment'] == 'Positive').tolist()

# tokenize, convert to sequences, and pad
# note: using the same padding for factual/counterfactual dataset pairs 
#       not sure on this for val/test
train_sequences = tokenizer.texts_to_sequences(train_texts)
cf_train_sequences = tokenizer.texts_to_sequences(cf_train_texts)
train_padding = max([len(i) for i in train_sequences] + 
                    [len(j) for j in cf_train_sequences])
train_data = pad_sequences(train_sequences, maxlen=train_padding, padding='post')
cf_train_data = pad_sequences(cf_train_sequences, maxlen=train_padding, padding='post')

val_sequences = tokenizer.texts_to_sequences(val_texts)
cf_val_sequences = tokenizer.texts_to_sequences(cf_val_texts)
val_padding = max([len(i) for i in val_sequences] + 
                  [len(j) for j in cf_val_sequences])
val_data = pad_sequences(val_sequences, maxlen=val_padding, padding='post')
cf_val_data = pad_sequences(cf_val_sequences, maxlen=val_padding, padding='post')

test_sequences = tokenizer.texts_to_sequences(test_texts)
cf_test_sequences = tokenizer.texts_to_sequences(cf_test_texts)
test_padding = max([len(i) for i in test_sequences] + 
                   [len(j) for j in cf_test_sequences])
test_data = pad_sequences(test_sequences, maxlen=test_padding, padding='post')
cf_test_data = pad_sequences(cf_test_sequences, maxlen=test_padding, padding='post')

# # Iterating with IDs 

# In[15]:


batch_size = 32
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

def get_dataloader(data, labels, batch_size):
    # Returns batch_size chunks of (encoded text, ID of text, label of text)
    batches = []
    for i in range(0, len(data), batch_size):
        text_tensor = torch.tensor(data[i:i + batch_size], device=device, dtype=torch.long)
        length_tensor = torch.tensor([len(j) for j in data[i:i+batch_size]], device=device)
        labels_tensor = torch.tensor(labels[i:i + batch_size], device=device, dtype=torch.float)
        batches.append((text_tensor, length_tensor, labels_tensor))
    return batches

def get_cf_dataloader(data, data_IDs, cf_data, cf_IDs, labels, batch_size):
    # Returns batch_size chunks of (encoded text, ID of text, label of text)
    batches = []
    for i in range(0, len(data), batch_size):
        text_tensor = torch.tensor(data[i:i + batch_size], device=device, dtype=torch.long)
        length_tensor = torch.tensor([len(j) for j in data[i:i+batch_size]], device=device)
        labels_tensor = torch.tensor(labels[i:i + batch_size], device=device, dtype=torch.float)
        
        cf_indices = [cf_IDs.index(data_ID) for data_ID in data_IDs[i:i + batch_size]]
        cf_text_tensor = torch.tensor(cf_data[cf_indices], device=device, dtype=torch.long)
        cf_length_tensor = torch.tensor([len(j) for j in cf_data[cf_indices]], device=device)
        
        batches.append((text_tensor, length_tensor, cf_text_tensor, cf_length_tensor, labels_tensor))
    return batches

train_loader = get_cf_dataloader(train_data, train_IDs, cf_train_data, cf_train_IDs, train_labels, batch_size)
val_loader = get_dataloader(val_data, val_labels, batch_size)
# val_loader = get_cf_dataloader(val_data, val_IDs, cf_val_data, cf_val_IDs, val_labels, batch_size)
test_loader = get_dataloader(test_data, test_labels, batch_size)
# test_loader = get_cf_dataloader(test_data, test_IDs, cf_test_data, cf_test_IDs, test_labels, batch_size)
print(len(train_loader), len(val_loader), len(test_loader))

# In[16]:

destination_folder = "."
lambda_coef = 0.01
criterion = torch.nn.BCELoss()

def clp_loss(criterion, output, labels, cf_output, lambda_coef):
    counterfactual_loss = (output - cf_output).abs().sum()
    sigmoid_out = torch.sigmoid(output)
    #print("Loss function ", sigmoid_out)
    loss = criterion(sigmoid_out, labels) - lambda_coef * counterfactual_loss
    #loss = criterion(output, labels) - lambda_coef * counterfactual_loss
    return loss


# In[18]:
#for text, cf_text, labels in train_loader:
    
    # get factual predictions
    #labels = labels.to(device)
#    output = torch.ones(labels.size())  # placeholder for model.predict(text)
    
    # get counterfactual predictions
#    cf_output = torch.ones(labels.size())  # placeholder for model.predict(cf_text)
    
    # compute CLP loss
#    loss = clp_loss(criterion, output, labels, cf_output, lambda_coef)
    
#    print(loss)


# In[ ]:

# Training Function

def train(model,
          optimizer,
          criterion = criterion,#nn.BCELoss(),
          #train_loader = train_iter,
          train_loader = train_loader,
          train_batches = len(train_loader), #train_batches,
          #valid_loader = valid_iter,
          valid_loader = val_loader,
          valid_batches = len(val_loader),#val_batches,
          num_epochs = 5,
          eval_every = len(train_loader) // 2, #len(train_iter) // 2,
          file_path = destination_folder,
          best_valid_loss = float("Inf")):
    
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    model.train()
    for epoch in range(num_epochs):
        #train_batches = 0
        for text, text_len, cf_text, cf_text_len, labels in train_loader:           
        #for (labels, (text, text_len)), _ in train_loader:           
        #for (labels, (title, title_len), (text, text_len), (titletext, titletext_len)), _ in train_loader:           
            #train_batches += 1
            #print("Entered train loop")
            labels = labels.to(device)
            #titletext = titletext.to(device)
            text = text.to(device)
            cf_text = cf_text.to(device)
            #titletext_len = titletext_len.to(device)
            #text_len = torch.ones((batch_size, ), dtype=torch.long, device=device) * train_padding 
            cf_output = model(cf_text, cf_text_len)
            output = model(text, text_len)
            loss = clp_loss(criterion, output, labels, cf_output, lambda_coef)
            #loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():                    
                  # validation loop
                  #val_batches = 0
                  for text, text_len, labels in valid_loader:           
                  #for text, text_len, labels in valid_loader:           
                  #for (labels, (text, text_len)), _ in valid_loader:           
                  #for (labels, (title, title_len), (text, text_len), (titletext, titletext_len)), _ in valid_loader:
                      labels = labels.to(device)
                      #titletext = titletext.to(device)
                      text = text.to(device)
                      #text_len = torch.ones((batch_size, ), dtype=torch.long, device=device) * val_padding
                      #titletext_len = titletext_len.to(device)
                      output = model(text, text_len)
                      output = torch.sigmoid(output)
                      loss = criterion(output, labels)
                      valid_running_loss += loss.item()
                      #val_batches += 1
                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / valid_batches#(val_data.shape[0]//batch_size)#(val_batches*batch_size)#len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0                
                valid_running_loss = 0.0
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*train_batches,#train_data.shape[0],#train_batches*batch_size,#len(train_loader),
                              average_train_loss, average_valid_loss))
                
                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + '/cf-model.pt', model, optimizer, best_valid_loss)
                    save_metrics(file_path + '/cf-metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    
    save_metrics(file_path + '/cf-metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')


model = LSTM().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

train(model=model, optimizer=optimizer, num_epochs=20)
train_loss_list, valid_loss_list, global_steps_list = load_metrics(destination_folder + '/cf-metrics.pt')

# Evaluation Function

def evaluate(model, test_loader, version='title', threshold=0.5):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for text, text_len, labels in test_loader:           
        #for text, text_len, labels in test_loader:           
        #for (labels, (text, text_len)), _ in test_loader:           
        #for (labels, (title, title_len), (text, text_len), (titletext, titletext_len)), _ in test_loader:           
            labels = labels.to(device)
            #titletext = titletext.to(device)
            text = text.to(device)
            #text_len = torch.ones((batch_size, )) * test_padding
            #titletext_len = titletext_len.to(device)
            output = model(text, text_len)
            
            sigmoid_out = torch.sigmoid(output)
            output = (sigmoid_out > threshold).int()
            #output = (output > threshold).int()
            y_pred.extend(output.tolist())
            y_true.extend(labels.tolist())
    
    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1,0], digits=4))
    
best_model = LSTM().to(device)
optimizer = optim.Adam(best_model.parameters(), lr=0.001)

load_checkpoint(destination_folder + '/cf-model.pt', best_model, optimizer)
evaluate(best_model, test_loader)
#evaluate(best_model, test_iter)

