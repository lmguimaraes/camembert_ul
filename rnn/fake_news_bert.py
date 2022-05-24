import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from transformers.models.camembert.tokenization_camembert import CamembertTokenizer

# Loading some sklearn packaces for modelling.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics import f1_score, accuracy_score
from wordcloud import WordCloud, STOPWORDS
from collections import Counter, defaultdict
from PIL import Image

import random
import warnings
import time
import datetime

from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler

stop = set(stopwords.words('french'))
plt.style.use('fivethirtyeight')
sns.set(font_scale=1.5)
pd.options.display.max_columns = 250
pd.options.display.max_rows = 250
warnings.filterwarnings('ignore')


#Setting seeds for consistent results.
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

def _classifier():
    if torch.cuda.is_available():        
        device = torch.device('cuda')    
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))  
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device('cpu')

    ##########################################
    #faire l'entrainement ici, ça va avoir l'air de ça

    train = "lire_donnees"
    test = "lire_donnees"

    labels = train['label'].values
    idx = len(labels)
    combined = pd.concat([train, test])
    combined = combined.fillna('no data')
    df = combined['title'] + ' ' + combined['author']
    combined = df.values

    ############################

    print('Original: ', combined[0])
    print('Tokenized: ', tokenizer.tokenize(combined[0]))
    print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(combined[0])))

    max_len = 0
    for text in combined:
        input_ids = tokenizer.encode(text, add_special_tokens=True)
        max_len = max(max_len, len(input_ids))
        
    print('Max sentence length: ', max_len)

    token_lens = []

    for text in combined:
        tokens = tokenizer.encode(text, max_length = 512)
        token_lens.append(len(tokens))

    train= combined[:idx]
    test = combined[idx:]
    train.shape

    input_ids, attention_masks, labels = tokenize_map(train, labels)
    test_input_ids, test_attention_masks= tokenize_map(test)

    # Combiner les inputs
    dataset = TensorDataset(input_ids, attention_masks, labels)
    # 80-20 train validation split.
    # Calculer nombre de samples
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))
    
    #Batch size pour entraînement, so we specify it here.La doc de BERT recommande 16 à 32 .
    batch_size = 16

    train_dataloader = DataLoader(
                train_dataset,  # The training samples.
                sampler = RandomSampler(train_dataset), # Select batches randomly
                batch_size = batch_size # Trains with this batch size.
            )
    validation_dataloader = DataLoader(
                val_dataset, # The validation samples.
                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
        )
    prediction_data = TensorDataset(test_input_ids, test_attention_masks)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)    

    # BertForSequenceClassification, the pretrained BERT model avec une seule couche. 

    model = BertForSequenceClassification.from_pretrained(
        'camembert-base', # 124-layer, 1024-hidden, 16-heads, 340M parameters 
        num_labels = 2, # 2 output pour 2 possibilités de classification 
        output_attentions = False, 
        output_hidden_states = False,
    )
    model.to(device)

    params = list(model.named_parameters())

    print('Le CamemBERT  a {:} paramètres differents.\n'.format(len(params)))

    print('==== Embedding Layer ====\n')

    for p in params[0:5]:
        print('{:<55} {:>12}'.format(p[0], str(tuple(p[1].size()))))

    print('\n==== First Transformer ====\n')

    for p in params[5:21]:
        print('{:<55} {:>12}'.format(p[0], str(tuple(p[1].size()))))

    print('\n==== Output Layer ====\n')

    for p in params[-4:]:
        print('{:<55} {:>12}'.format(p[0], str(tuple(p[1].size()))))

    # Note: AdamW est huggingface pas pytorch.
    # Le 'W' est 'Weight Decay fix' je crois...

    optimizer = AdamW(model.parameters(),
                    lr = 6e-6, # args.learning_rate
                    eps = 1e-8 # args.adam_epsilon
                    )
    # Training epochs. La doc de BERT recommende entre 2 et 4. 
    epochs = 3
    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    # Ce code est basé sur le `run_glue.py` script:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128
    training_stats = []
    total_t0 = time.time()
    for epoch_i in range(0, epochs):
        
        # ========================================
        #               Training
        # ========================================
        
        # Une passage sur l'ensemble d'entraînement.

        print('')
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        
        t0 = time.time()
        total_train_loss = 0

        # model.train() change le mode il NE FAIT PAS l'entraînement
        
        # `dropout` and `batchnorm` layers behave differently during training vs. test ,
        # source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch
        
        model.train()

        for step, batch in enumerate(train_dataloader):

            # MÀJ à chaque 10 batch
            if step % 50 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            
            b_input_ids = batch[0].to(device).to(torch.int64)
            b_input_mask = batch[1].to(device).to(torch.int64)
            b_labels = batch[2].to(device).to(torch.int64)
            
            # Nettoyer les gradients avant le backward pass. PyTorch ne le fait pas automatiquement. 
            # Source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
            
            model.zero_grad()        

            # Fait le forward pass.
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers BertForSequenceClassification.
            loss = model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels)[0]
            logits = model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels)[1]
            #print(loss)

            # Accumulez la perte d'apprentissage sur tous les lots afin de pouvoir calculer la perte moyenne à la fin, 
            # `loss` c'est un tensor avec une seule valeur; le `.item()` function retourne la valeur Python du tensor.
            
            total_train_loss += loss.item()

            # Backward pass pour calculer les gradients.
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()

        # Calculer la moyenne des pertes
        
        avg_train_loss = total_train_loss / len(train_dataloader)            
        
        training_time = format_time(time.time() - t0)

        print('')
        print('  Average training loss: {0:.2f}'.format(avg_train_loss))
        print('  Training epcoh took: {:}'.format(training_time))
            
        # ========================================
        #               Validation
        # ========================================
        # Après l'achèvement de chaque époque de formation, mesurez notre performance sur notre ensemble de validation.

        print('')
        print('Running Validation...')

        t0 = time.time()

        # Change au mode eval
        
        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0
        total_eval_f1 = 0
        nb_eval_steps = 0
        
        for batch in validation_dataloader:
            
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
                        
            with torch.no_grad():        

                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the 'segment ids', which differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is down here: 
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers BertForSequenceClassification.
                # Get the 'logits' output by the model. The 'logits' are the output values prior to applying an activation function like the softmax.
                
                loss = model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels)[0]

                logits = model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels)[1]
            
            total_eval_loss += loss.item()
            
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            total_eval_accuracy += flat_accuracy(logits, label_ids)
            total_eval_f1 += flat_f1(logits, label_ids)
        
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print('  Accuracy: {0:.2f}'.format(avg_val_accuracy))
        
        avg_val_f1 = total_eval_f1 / len(validation_dataloader)
        print('  F1: {0:.2f}'.format(avg_val_f1))
        
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        
        validation_time = format_time(time.time() - t0)
        
        print('  Validation Loss: {0:.2f}'.format(avg_val_loss))
        print('  Validation took: {:}'.format(validation_time))
        
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Val_F1' : avg_val_f1,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print('')
    print('Training complete!')

    print('Total training took {:} (h:mm:ss)'.format(format_time(time.time()-total_t0)))

    pd.set_option('precision', 3)

    df_stats = pd.DataFrame(data=training_stats)
    df_stats = df_stats.set_index('epoch')

    # Afficher la table

    df_stats

    #Plot pour visualiser, pas vraiment necessaire
    ###############################################################
    fig, axes = plt.subplots(figsize=(12,8))

    plt.plot(df_stats['Training Loss'], 'b-o', label='Training')
    plt.plot(df_stats['Valid. Loss'], 'g-o', label='Validation')

    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()
    plt.xticks([1, 2])

    plt.show()
    ################################################################

    print('Predicting labels for {:,} test sentences...'.format(len(test_input_ids)))
    model.eval()

    predictions = []

    for batch in prediction_dataloader:

        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, = batch
    
        with torch.no_grad():         
            outputs = model(b_input_ids, token_type_ids=None, 
                            attention_mask=b_input_mask)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        predictions.append(logits)
    print('DONE.')
    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    #result = pd.read_csv('../input/fake-news/submit.csv')
    #result['label'] = flat_predictions
    #result.head(10)

def tokenize_map(sentence,labs='None'):
    global labels
    
    input_ids = []
    attention_masks = []
    
    for text in sentence:        
        encoded_dict = tokenizer.encode_plus(
                            text,                      
                            add_special_tokens = True, # Ajouter '[CLS]' et '[SEP]'
                            truncation='longest_first', 
                            max_length = 84,           #Choisir le max length
                            pad_to_max_length = True, 
                            return_attention_mask = True,
                            return_tensors = 'pt',   
                       )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    
    if labs != 'None': 
        labels = torch.tensor(labels)
        return input_ids, attention_masks, labels
    else:
        return input_ids, attention_masks

def flat_accuracy(preds, labels): 
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
        
    return accuracy_score(labels_flat, pred_flat)

def flat_f1(preds, labels):  
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
        
    return f1_score(labels_flat, pred_flat)

def format_time(elapsed):    
    elapsed_rounded = int(round((elapsed)))
    
    return str(datetime.timedelta(seconds=elapsed_rounded))