# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 05:18:36 2022
taudata Library for snScrape twitter scraper
@author: Taufik Sutanto
"""
import warnings; warnings.simplefilter('ignore')
import torch, numpy as np, sys
import datetime, time, pandas as pd
import matplotlib.pyplot as plt, ast
import tauData as tau
import seaborn as sns;sns.set(style='darkgrid')
from tqdm import tqdm
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW#, BertConfig
from transformers import get_linear_schedule_with_warmup
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

def genderHeuristic(dff, dfG):
    pria = "bang bung bapak pak muhammad muhamad ahmad bambang tono raden \
    fajar pakde paman anto ridwan ayah abi anton agus yusuf tarto roby iwan boy \
        sutrisno slamet mulyadi herman supardi ismail suprianto suparman junaidi wahyudi".split()
    wanita = "dewi ayu ratna wati putri jeng susan bunda nissa cintya indah lala brigitta emak \
        nurhayati sulastri sumiati sri wahyuni sumarni sunarti siti aminah ernawati kartini".split()
    media = "tv news online post kota harian fm radio tempo tribun kompas detik berita tirtoid solopos \
        antara media polda polsek dpc dpp liputan times jpnn cnn coid harian dotco cnbc info \
        partai pks pdi golkar gerindra pkb nasdem demokrat hanura pkb \
            psi perindo ppp pbb detik tribun cnn teropongrakyat".split()
    unknown = "photo php groups reel".split()
    df = dff.copy()
    count_ = 0
    for i, d in tqdm(df.iterrows()):
        try:
            p = ast.literal_eval(d.pagemap)
            try:
                df.at[i,'username'] = p['person'][0]['additionalname'][:32]
            except:
                pass
            try:
                df.at[i,'name'] = p['person'][0]['givenname'][:255]
            except:
                pass
        except:
            pass
        if pd.isna(d.gender):
            check = True
            if df.at[i,'name']:
                nama = str(d.username) + " " + str(df.at[i,'name'])
            else:
                nama = str(d.username)
                
            if len(nama)<5 or str(d.username).isdigit() or tau.noVocal(str(d.username)):
                df.at[i,'gender'] = "Unknown/Hidden"
                check = False
                
            if check and d.username:
                try:
                    df.at[i,'gender'] = dfG[dfG['username']==d.username]["Gender"].values[0]
                    check = False
                except:
                    pass
                
            if check:
                for u in unknown:
                    if u in nama:
                        df.at[i,'gender'] = "Unknown/Hidden"
                        check = False
                        break
            if check:        
                for m in media:
                    if m in nama:
                        df.at[i,'gender'] = "MediaMasa/Institusi"
                        check = False
                        break
            if check:
                for w in wanita:
                    if w in nama:
                        df.at[i,'gender'] = "Wanita"
                        check = False
                        break
            if check:
                for p in pria:
                    if p in nama:
                        df.at[i,'gender'] = "Pria"
                        check = False
                        break
            if not check:
                count_ += 1
    return df, count_

def loadBertSentimen():
    pretrained= "mdhugol/indonesia-bert-sentiment-classification"
    model = AutoModelForSequenceClassification.from_pretrained(pretrained)
    tokenizer = AutoTokenizer.from_pretrained(pretrained)    
    return tokenizer, model
    
def sentiment(text, tokenizer, model):
    sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    label_index = {'LABEL_0': 1, 'LABEL_1': 0, 'LABEL_2': -1}
    result = sentiment_analysis(text)
    status = label_index[result[0]['label']]
    score = result[0]['score']
    return status, score

def prediction(txt, num2label, device, tokenizer, model):
    encoding = tokenizer.encode_plus(
                        txt,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 256,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                        truncation=True, ).to(device)
    seqOutput = model(encoding["input_ids"], encoding["attention_mask"])
    probabilities = seqOutput.logits.detach().cpu().numpy()
    probabilities = F.softmax(torch.tensor(probabilities[0]), dim=0)
    bidang = num2label[int(np.argmax(probabilities, axis=0).flatten()[0].cpu().numpy())]
    return bidang

def gpuTest():
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def trainModel(train_sentence, train_labels, device, num_labels, train_size=0.85, epochs=30, model_max_length=256, batch_size=32, output_dir="models/"):
    # Tokenizing
    print("Loading BertTokenizer ... ")
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', model_max_length=model_max_length)
    tokenizer.model_max_length = sys.maxsize
    
    print("Preparing Training and Validation Samples ... ")
    sent_length = []
    for sent in train_sentence:
        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        input_ids = tokenizer.encode(sent, add_special_tokens=True)
        sent_length.append(len(input_ids))
        
    print('Average length = ', sum(sent_length)/len(sent_length))
    #print('Median length = ', statistics.median(sent_length))
    
    # Tokenize all of the sentences and map the tokens to their word IDs.
    input_ids = []
    attention_masks = []    
    for sent in train_sentence:
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = model_max_length,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                            truncation=True,
                       )
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    train_labels = torch.tensor(train_labels)
    
    # Optimizing parameter, starting by more split
    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, train_labels)
    
    # Create a train-validation split.
    # Calculate the number of samples to include in each set.
    train_size = int(train_size * len(dataset))
    val_size = len(dataset) - train_size
    
    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))

    # For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32.
    batch_size = batch_size
    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order. 
    train_dataloader = DataLoader(
                train_dataset,  # The training samples.
                sampler = RandomSampler(train_dataset), # Select batches randomly
                batch_size = batch_size, # Trains with this batch size.
                num_workers=4
            )
    
    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
                val_dataset, # The validation samples.
                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size = batch_size, # Trains with this batch size.
                num_workers=4 # Evaluate with this batch size.
            )
    
    # Load BertForSequenceClassification, the pretrained BERT model with a single 
    # linear classification layer on top. 
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-multilingual-cased', # Use the 12-layer BERT model, with an cased vocab.
        num_labels = num_labels, 
        output_attentions = False, # return attentions weights
        output_hidden_states = False, ) # returns all hidden-states

    # Tell pytorch to run this model on the GPU.
    model.to(device) # model.cuda()
    
    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())
    print('The BERT model has {:} different named parameters.\n'.format(len(params)))
    print('==== Embedding Layer ====\n')
    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    print('\n==== First Transformer ====\n')
    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    print('\n==== Output Layer ====\n')
    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
        
    # Optimizer & Learning Rate Scheduler
    optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8 )
    # Total number of training steps is [number of batches] x [number of epochs]. 
    # (Note that this is not the same as the number of training samples).
    print('Jumlah batch :', len(train_dataloader))
    total_steps = len(train_dataloader) * epochs
    
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    
    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128
    
    # List variable for store training and validation loss, validation accuracy, and timings.
    training_stats = []
    
    # Measure the total training time for the whole run.
    total_t0 = time.time()
    
    # For each epoch...
    early_stopper = EarlyStopper(patience=3, min_delta=.01)
    print("Model in GPU = ", next(model.parameters()).is_cuda)
    for epoch_i in range(0, epochs):
        # ========================================
        #               Training
        # ========================================
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        # Measure how long the training epoch takes.
        t0 = time.time()
        # Reset the total loss for this epoch.
        total_train_loss = 0
        # Put the model into training mode
        model.train()
    
        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            # Progress update every 20 batches.
            if step % 20 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
    
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
    
            # Always clear any previously calculated gradients before performing a backward pass
            model.zero_grad()        
            # Perform a forward pass (evaluate the model on this training batch).
            # token_type_ids is same as the "segment ids", which differentiates 
            # sentence 1 and 2 in sentence-pair tasks
            loss, logits = model(b_input_ids, 
                                 token_type_ids=None,
                                 attention_mask=b_input_mask, 
                                 labels=b_labels, return_dict=False)
    
            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. 
            total_train_loss += loss.item()
            # Perform a backward pass to calculate the gradients.
            loss.backward()
            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update parameters and take a step using the computed gradient.
            optimizer.step()
            # Update the learning rate.
            scheduler.step()
    
        
        avg_train_loss = total_train_loss / len(train_dataloader) # Calculate the average loss over all of the batches.           
        training_time = format_time(time.time() - t0) # Measure how long this epoch took
        print("\n  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.
        print("")
        print("Running Validation...")
    
        t0 = time.time()
        model.eval() # Put the model in evaluation mode (batchnorm, dropout disable)
    
        # Tracking variables 
        total_eval_accuracy = 0
        total_eval_loss = 0
    
        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            # Deactivate autograd, it will reduce memory usage and speed up computations
            # but you won’t be able to backprop (which you don’t want in an eval script).
            with torch.no_grad():        
    
                # Forward pass, calculate logit predictions.
                (loss, logits) = model(b_input_ids, 
                                       token_type_ids=None, 
                                       attention_mask=b_input_mask,
                                       labels=b_labels, return_dict=False)
                
            # Accumulate the validation loss.
            total_eval_loss += loss.item()
    
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
    
            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)
            
        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
    
        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))
        # Record all statistics from this epoch.
        training_stats.append(
            {
                'Epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Validation Loss': avg_val_loss,
                'Validation Accuracy': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )
        if early_stopper.early_stop(avg_val_loss):             
            break
    print("\nTraining complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    saveModel(tokenizer, model, output_dir=output_dir)
    #pd.set_option('precision', 2)
    fStats = 'data/df_stats.csv'
    df_stats = pd.DataFrame(data=training_stats) # Create a DataFrame from our training statistics.
    df_stats = df_stats.set_index('Epoch') # Use the 'epoch' as the row index.
    df_stats.to_csv(fStats, index=False, encoding='utf8')
    print(df_stats.head()) # Display the table.
    #visualize(df_stats)
    return tokenizer, model

def predict(test_sentences, test_labels, device, tokenizer, model, model_max_length=256, batch_size=32):
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []
    # For every sentence...
    for sent in test_sentences:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = model_max_length,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                            truncation=True,
                       )
        
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
    
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(test_labels)
    
    # Create the DataLoader.
    prediction_data = TensorDataset(input_ids, attention_masks, labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, 
                                       batch_size=batch_size, num_workers=4)
    
    # Prediction on test set
    print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))
    model.eval() # Put model in evaluation mode
    predictions , true_labels = [], [] # Tracking variables  
    for batch in prediction_dataloader:
      batch = tuple(t.to(device) for t in batch) # Add batch to GPU
      b_input_ids, b_input_mask, b_labels = batch # Unpack the inputs from our dataloader
      
      # Telling the model not to compute or store gradients, saving memory and speeding up prediction
      with torch.no_grad():
          outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask) # Forward pass, calculate logit predictions
    
      logits = outputs[0]
      logits = logits.detach().cpu().numpy()# Move logits and labels to CPU
      label_ids = b_labels.to('cpu').numpy()
      
      # Store predictions and true labels
      predictions.append(logits)
      true_labels.append(label_ids)
    print('    DONE.')
    return predictions, true_labels
    

def saveModel(tokenizer, model, output_dir="models/"):
    print("Saving model to %s" % output_dir, end = '', flush=True)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(" Finished!", flush=True)
    return True

def loadModel(device, BertForSequenceClassification, BertTokenizer, output_dir="models/"):
    print("Loading model from %s" % output_dir, end = '', flush=True)
    # Load a trained model and vocabulary that you have fine-tuned
    model = BertForSequenceClassification.from_pretrained(output_dir)
    tokenizer = BertTokenizer.from_pretrained(output_dir)
    # Copy the model to the GPU.
    model.to(device)
    print(" Finished!", flush=True)
    return tokenizer, model 

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def visualize(df_stats, figSize=(12,6)):
    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = figSize
    # Plot the learning curve.
    plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
    plt.plot(df_stats['Validation Loss'], 'g-o', label="Validation")
    # Label the plot.
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks(list(range(1,df_stats.shape[0]+1)))
    plt.show()

if __name__ == '__main__':
    pass