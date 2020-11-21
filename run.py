import pandas as pd
import torch

# Preliminaries

from torchtext.data import Field, RawField, TabularDataset, BucketIterator
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Models
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Training

import torch.optim as optim

# Evaluation

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#import seaborn as sns

# Fields

label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
text_field = Field(tokenize='spacy', lower=True, include_lengths=True, batch_first=True)
#unused_field = RawField()#lower=True, include_lengths=True, batch_first=True)
fields = [('label', label_field), ('Text', text_field)]
#fields = [('label', label_field), ('Sentiment', unused_field), ('Text', text_field), ('batch_id', unused_field)]

# TabularDataset

source_folder = "sentiment/combined/paired/"
destination_folder = "."
train, valid, test = TabularDataset.splits(path=source_folder, train='train_preprocessed.csv', validation='dev_preprocessed.csv', test='test_preprocessed.csv',
                                           format='CSV', fields=fields, skip_header=True)

print(train)
#exit(0)

# Iterators
device="cuda" 
print("Device ", device)
train_iter = BucketIterator(train, batch_size=32, sort_key=lambda x: len(x.Text),
                            device=device, sort=True, sort_within_batch=True)
valid_iter = BucketIterator(valid, batch_size=32, sort_key=lambda x: len(x.Text),
                            device=device, sort=True, sort_within_batch=True)
test_iter = BucketIterator(test, batch_size=32, sort_key=lambda x: len(x.Text),
                            device=device, sort=True, sort_within_batch=True)

text_field.build_vocab(train, min_freq=3)
print(len(text_field.vocab), len(train_iter), train_iter)
#for (label, (text, text_len)), _ in train_iter:
#    print(label, text)

# Vocabulary

### Possible simpler data loading

tokenizer = Tokenizer(num_words=len(text_field.vocab), oov_token=True)

train_df = pd.read_csv(source_folder+"train"+"_preprocessed.csv")
#train_df = train_df.sample(frac=1)
train_texts = list(train_df["Text"])
train_labels = list(train_df["label"])
tokenizer.fit_on_texts(train_texts)
train_sequences = tokenizer.texts_to_sequences(train_texts)
max_padding = max([len(i) for i in train_sequences])
train_data = pad_sequences(train_sequences, maxlen=max_padding, padding='post')
batch_size = 32

val_df = pd.read_csv(source_folder+"dev"+"_preprocessed.csv")
val_df = val_df.sample(frac=1)
val_texts = list(val_df["Text"])
val_labels = list(val_df["label"])
val_sequences = tokenizer.texts_to_sequences(val_texts)
val_padding = max([len(i) for i in val_sequences])
val_data = pad_sequences(val_sequences, maxlen=max_padding, padding='post')

test_df = pd.read_csv(source_folder+"test"+"_preprocessed.csv")
test_df = test_df.sample(frac=1)
test_texts = list(test_df["Text"])
test_labels = list(test_df["label"])
test_sequences = tokenizer.texts_to_sequences(test_texts)
max_padding = max([len(i) for i in test_sequences])
test_data = pad_sequences(test_sequences, maxlen=max_padding, padding='post')

print(train_data.shape)
print(val_data.shape)
print(test_data.shape)

def get_dataloader(data, labels, batch_size):
    # Returns batch_size chunks of (encoded texts, length of each text, label of each text)
    obj = []
    for i in range(0, len(data), batch_size):
        obj.append((torch.tensor(data[i:i+batch_size], device=device, dtype=torch.long), torch.tensor([len(j) for j in data[i:i+batch_size]], device=device), torch.tensor(labels[i:i+batch_size],device=device, dtype=torch.float)))
    return obj

train_dataloader = get_dataloader(train_data, train_labels, batch_size)
val_dataloader = get_dataloader(val_data, val_labels, batch_size)
test_dataloader = get_dataloader(test_data, test_labels, batch_size)

train_batches = 0
for text, text_len, labels in train_dataloader:
    train_batches+=1

val_batches = 0
for text, text_len, labels in val_dataloader:
    val_batches+=1

test_batches = 0
for text, text_len, labels in test_dataloader:
    test_batches+=1

print(len(train_iter), train_batches, len(test_iter), test_batches)

class LSTM(nn.Module):

    def __init__(self, dimension=128):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(len(text_field.vocab), 300)
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=300,
                            hidden_size=dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.5)

        self.fc = nn.Linear(2*dimension, 1)

    def forward(self, text, text_len):

        text_emb = self.embedding(text)
        #output = self.lstm(text_emb)
        packed_input = pack_padded_sequence(text_emb, text_len.cpu(), batch_first=True, enforce_sorted=False)
        #packed_input = pack_padded_sequence(text_emb, text_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.drop(out_reduced)

        text_fea = self.fc(text_fea)
        text_fea = torch.squeeze(text_fea, 1)
        text_out = torch.sigmoid(text_fea)

        return text_out


# Save and Load Functions

def save_checkpoint(save_path, model, optimizer, valid_loss):

    if save_path == None:
        return
    
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'valid_loss': valid_loss}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model, optimizer):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    
    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):

    if save_path == None:
        return
    
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']

# Training Function

def train(model,
          optimizer,
          criterion = nn.BCELoss(),
          #train_loader = train_iter,
          train_loader = train_dataloader,
          train_batches = train_batches,
          #valid_loader = valid_iter,
          valid_loader = val_dataloader,
          valid_batches = val_batches,
          num_epochs = 5,
          eval_every = train_batches // 2, #len(train_iter) // 2,
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
        for text, text_len, labels in train_loader:           
        #for (labels, (text, text_len)), _ in train_loader:           
        #for (labels, (title, title_len), (text, text_len), (titletext, titletext_len)), _ in train_loader:           
            #train_batches += 1
            #print("Entered train loop")
            labels = labels.to(device)
            #titletext = titletext.to(device)
            text = text.to(device)
            #titletext_len = titletext_len.to(device)
            output = model(text, text_len)

            loss = criterion(output, labels)
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
                  #for (labels, (text, text_len)), _ in valid_loader:           
                  #for (labels, (title, title_len), (text, text_len), (titletext, titletext_len)), _ in valid_loader:
                      labels = labels.to(device)
                      #titletext = titletext.to(device)
                      text = text.to(device)
                      #titletext_len = titletext_len.to(device)
                      output = model(text, text_len)

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
                    save_checkpoint(file_path + '/model.pt', model, optimizer, best_valid_loss)
                    save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    
    save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')


model = LSTM().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

train(model=model, optimizer=optimizer, num_epochs=20)
train_loss_list, valid_loss_list, global_steps_list = load_metrics(destination_folder + '/metrics.pt')

# Evaluation Function

def evaluate(model, test_loader, version='title', threshold=0.5):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for text, text_len, labels in test_loader:           
        #for (labels, (text, text_len)), _ in test_loader:           
        #for (labels, (title, title_len), (text, text_len), (titletext, titletext_len)), _ in test_loader:           
            labels = labels.to(device)
            #titletext = titletext.to(device)
            text = text.to(device)
            #titletext_len = titletext_len.to(device)
            output = model(text, text_len)

            output = (output > threshold).int()
            y_pred.extend(output.tolist())
            y_true.extend(labels.tolist())
    
    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1,0], digits=4))
    
best_model = LSTM().to(device)
optimizer = optim.Adam(best_model.parameters(), lr=0.001)

load_checkpoint(destination_folder + '/model.pt', best_model, optimizer)
evaluate(best_model, test_dataloader)
#evaluate(best_model, test_iter)

