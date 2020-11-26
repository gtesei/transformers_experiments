import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import copy
import torch.optim as optim
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AdamW , get_linear_schedule_with_warmup
from datasets import load_dataset, load_metric
from sklearn.metrics import accuracy_score  , classification_report

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import psutil
import humanize
import os
import GPUtil as GPU

def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def printm():
    GPUs = GPU.getGPUs()
    process = psutil.Process(os.getpid())
    print("Gen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), " | Proc size: " + humanize.naturalsize( process.memory_info().rss))
    if len(GPUs) > 0:
        gpu = GPUs[0]
        print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f} MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))
    else:
        print("NO GPU")
        
printm()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device::",device)
SEED = 1
set_seed(SEED)

############ Parameters 

DATASET = 'coliee2020'# 'mrpc' # coliee2020
bert_model = "albert-base-v2"  # 'albert-base-v2', 'albert-large-v2', 'albert-xlarge-v2', 'albert-xxlarge-v2', 'bert-base-uncased', ...
freeze_bert = False  # if True, freeze the encoder weights and only update the classification layer weights
maxlen = 128  # maximum length of the tokenized input sentence pair : if greater than "maxlen", the input is truncated and else if smaller, the input is padded
bs = 16  # batch size
iters_to_accumulate = 2  # the gradient accumulation adds gradients over an effective batch of size : bs * iters_to_accumulate. If set to "1", you get the usual batch size
lr = 2e-5  # learning rate
epochs = 4  # number of training epochs

############ Loading the dataset
def load_dataframes(dataset='mrpc'):
    if dataset=='mrpc':
        print(">>> Loading glue / mrpc")
        dataset = load_dataset('glue', 'mrpc')
        split = dataset['train'].train_test_split(test_size=0.1, seed=1)  # split the original training data for validation
        train = split['train']  # 90 % of the original training data
        val = split['test']   # 10 % of the original training data
        test = dataset['validation']  # the original validation data is used as test data because the test labels are not available with the datasets library
        
        # Transform data into pandas dataframes
        df_train = pd.DataFrame(train)
        df_val = pd.DataFrame(val)
        df_test = pd.DataFrame(test)
    elif dataset=='coliee2020':
        all_data = pd.read_csv('coliee2020_train_as_csv.csv') 
        all_data['new_label'] = np.where(all_data['label']=='Y',1,0)
        del all_data['label'] , all_data['article_number'] , all_data['id']
        all_data.rename(columns={'t1':'sentence1', 't2':'sentence2', 'new_label':'label'},inplace=True)
        df_train , df_val , df_test = np.split(all_data.sample(frac=1, random_state=SEED), [ int(0.6*len(all_data)), int(0.8*len(all_data)) ] )
        df_train.reset_index(drop=True,inplace=True)
        df_val.reset_index(drop=True,inplace=True)
        df_test.reset_index(drop=True,inplace=True)
    else:
        raise Exception("Dataset not supported::"+str(dataset))

    ##
    return df_train , df_val, df_test 

df_train , df_val, df_test = load_dataframes(dataset=DATASET)

print("df_train:",df_train.shape)
print("df_val:",df_val.shape)
print("df_test:",df_test.shape)

print(df_train.head())

class CustomDataset(Dataset):

    def __init__(self, data, maxlen, with_labels=True, bert_model='albert-base-v2'):

        self.data = data  # pandas dataframe
        #Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)  

        self.maxlen = maxlen
        self.with_labels = with_labels 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # Selecting sentence1 and sentence2 at the specified index in the data frame
        sent1 = str(self.data.loc[index, 'sentence1'])
        sent2 = str(self.data.loc[index, 'sentence2'])

        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer(sent1, sent2, 
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,  # Truncate to max_length
                                      max_length=self.maxlen,  
                                      return_tensors='pt')  # Return torch.Tensor objects
        
        token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
        attn_masks = encoded_pair['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        if self.with_labels:  # True if the dataset has labels
            label = self.data.loc[index, 'label']
            return token_ids, attn_masks, token_type_ids, label  
        else:
            return token_ids, attn_masks, token_type_ids
        
################## SentencePairClassifier
class SentencePairClassifier(nn.Module):

    def __init__(self, bert_model="albert-base-v2", freeze_bert=False):
        super(SentencePairClassifier, self).__init__()
        #  Instantiating BERT-based model object
        self.bert_layer = AutoModel.from_pretrained(bert_model)

        #  Fix the hidden-state size of the encoder outputs (If you want to add other pre-trained models here, search for the encoder output size)
        if bert_model == "albert-base-v2":  # 12M parameters
            hidden_size = 768
        elif bert_model == "albert-large-v2":  # 18M parameters
            hidden_size = 1024
        elif bert_model == "albert-xlarge-v2":  # 60M parameters
            hidden_size = 2048
        elif bert_model == "albert-xxlarge-v2":  # 235M parameters
            hidden_size = 4096
        elif bert_model == "bert-base-uncased": # 110M parameters
            hidden_size = 768

        # Freeze bert layers and only train the classification layer weights
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # Classification layer
        self.cls_layer = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(p=0.1)

    @autocast()  # run in mixed precision
    def forward(self, input_ids, attn_masks, token_type_ids):
        '''
        Inputs:
            -input_ids : Tensor  containing token ids
            -attn_masks : Tensor containing attention masks to be used to focus on non-padded values
            -token_type_ids : Tensor containing token type ids to be used to identify sentence1 and sentence2
        '''

        # Feeding the inputs to the BERT-based model to obtain contextualized representations
        cont_reps, pooler_output = self.bert_layer(input_ids, attn_masks, token_type_ids)

        # Feeding to the classifier layer the last layer hidden-state of the [CLS] token further processed by a
        # Linear Layer and a Tanh activation. The Linear layer weights were trained from the sentence order prediction (ALBERT) or next sentence prediction (BERT)
        # objective during pre-training.
        logits = self.cls_layer(self.dropout(pooler_output))

        return logits

def evaluate_loss(net, device, criterion, dataloader):
    net.eval()

    mean_loss = 0
    count = 0

    with torch.no_grad():
        for it, (seq, attn_masks, token_type_ids, labels) in enumerate(tqdm(dataloader)):
            seq, attn_masks, token_type_ids, labels = \
                seq.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)
            logits = net(seq, attn_masks, token_type_ids)
            mean_loss += criterion(logits.squeeze(-1), labels.float()).item()
            count += 1

    return mean_loss / count

print("Creation of the models' folder...")
if not os.path.exists('models'):
    os.makedirs('models')
    
############################### Training 
    
def train_bert(net, criterion, opti, lr, lr_scheduler, train_loader, val_loader, epochs, iters_to_accumulate):

    best_loss = np.Inf
    best_ep = 1
    nb_iterations = len(train_loader)
    print_every = nb_iterations // 5  # print the training loss 5 times per epoch
    iters = []
    train_losses = []
    val_losses = []

    scaler = GradScaler()

    for ep in range(epochs):

        net.train()
        running_loss = 0.0
        for it, (seq, attn_masks, token_type_ids, labels) in enumerate(tqdm(train_loader)):

            # Converting to cuda tensors
            seq, attn_masks, token_type_ids, labels = \
                seq.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)
    
            # Enables autocasting for the forward pass (model + loss)
            with autocast():
                # Obtaining the logits from the model
                logits = net(seq, attn_masks, token_type_ids)

                # Computing loss
                loss = criterion(logits.squeeze(-1), labels.float())
                loss = loss / iters_to_accumulate  # Normalize the loss because it is averaged

            # Backpropagating the gradients
            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            scaler.scale(loss).backward()

            if (it + 1) % iters_to_accumulate == 0:
                # Optimization step
                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, opti.step() is then called,
                # otherwise, opti.step() is skipped.
                scaler.step(opti)
                # Updates the scale for next iteration.
                scaler.update()
                # Adjust the learning rate based on the number of iterations.
                lr_scheduler.step()
                # Clear gradients
                opti.zero_grad()


            running_loss += loss.item()

            if (it + 1) % print_every == 0:  # Print training loss information
                print()
                print("Iteration {}/{} of epoch {} complete. Loss : {} "
                      .format(it+1, nb_iterations, ep+1, running_loss / print_every))

                running_loss = 0.0


        val_loss = evaluate_loss(net, device, criterion, val_loader)  # Compute validation loss
        print()
        print("Epoch {} complete! Validation Loss : {}".format(ep+1, val_loss))

        if val_loss < best_loss:
            print("Best validation loss improved from {} to {}".format(best_loss, val_loss))
            print()
            net_copy = copy.deepcopy(net)  # save a copy of the model
            best_loss = val_loss
            best_ep = ep + 1

    # Saving the model
    path_to_model='models/{}_{}_lr_{}_val_loss_{}_ep_{}.pt'.format(DATASET,bert_model, lr, round(best_loss, 5), best_ep)
    torch.save(net_copy.state_dict(), path_to_model)
    print("The model has been saved in {}".format(path_to_model))

    del loss
    torch.cuda.empty_cache()
    
    # return path to the best model 
    return path_to_model



# Creating instances of training and validation set
print("Reading training data...")
train_set = CustomDataset(df_train, maxlen, bert_model)
print("Reading validation data...")
val_set = CustomDataset(df_val, maxlen, bert_model)
# Creating instances of training and validation dataloaders
train_loader = DataLoader(train_set, batch_size=bs, num_workers=5)
val_loader = DataLoader(val_set, batch_size=bs, num_workers=5)



net = SentencePairClassifier(bert_model, freeze_bert=freeze_bert)

if torch.cuda.device_count() > 1:  # if multiple GPUs
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = nn.DataParallel(net)

net.to(device)

criterion = nn.BCEWithLogitsLoss()
opti = AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
num_warmup_steps = 0 # The number of steps for the warmup phase.
num_training_steps = epochs * len(train_loader)  # The total number of training steps
t_total = (len(train_loader) // iters_to_accumulate) * epochs  # Necessary to take into account Gradient accumulation
lr_scheduler = get_linear_schedule_with_warmup(optimizer=opti, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)

path_to_best_model = train_bert(net, criterion, opti, lr, lr_scheduler, train_loader, val_loader, epochs, iters_to_accumulate)

######################### Prediction
print("#####################################################################")
print("######################## TEST ACCURACY ##############################")
print("####################################################################")
    
def get_probs_from_logits(logits):
    """
    Converts a tensor of logits into an array of probabilities by applying the sigmoid function
    """
    probs = torch.sigmoid(logits.unsqueeze(-1))
    return probs.detach().cpu().numpy()

def test_prediction(net, device, dataloader, with_labels=True):
    """
    Predict the probabilities on a dataset with or without labels and print the result in a file
    """
    net.eval()
    #w = open(result_file, 'w')
    probs_all = []

    with torch.no_grad():
        if with_labels:
            for seq, attn_masks, token_type_ids, _ in tqdm(dataloader):
                seq, attn_masks, token_type_ids = seq.to(device), attn_masks.to(device), token_type_ids.to(device)
                logits = net(seq, attn_masks, token_type_ids)
                probs = get_probs_from_logits(logits.squeeze(-1)).squeeze(-1)
                probs_all += probs.tolist()
        else:
            for seq, attn_masks, token_type_ids in tqdm(dataloader):
                seq, attn_masks, token_type_ids = seq.to(device), attn_masks.to(device), token_type_ids.to(device)
                logits = net(seq, attn_masks, token_type_ids)
                probs = get_probs_from_logits(logits.squeeze(-1)).squeeze(-1)
                probs_all += probs.tolist()

    #w.writelines(str(prob)+'\n' for prob in probs_all)
    #w.close()
    return probs_all

###
print("Reading test data...")
test_set = CustomDataset(df_test, maxlen, bert_model)
test_loader = DataLoader(test_set, batch_size=bs, num_workers=5)

model = SentencePairClassifier(bert_model)
if torch.cuda.device_count() > 1:  # if multiple GPUs
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

print()
print("Loading the weights of the model...")
model.load_state_dict(torch.load(path_to_best_model))
model.to(device)

print("Predicting on test data...")
test_probs_list = test_prediction(net=model, device=device, dataloader=test_loader, with_labels=True)
test_pred = [1 if tt>= 0.5 else 0 for tt in test_probs_list]

test_accuracy = accuracy_score(test_pred,df_test['label'].tolist())
print("TEST accuracy:::", test_accuracy)
print()
print(classification_report(test_pred,df_test['label'].tolist()))
