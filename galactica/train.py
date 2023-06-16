# Imports
from importlib import reload
import os
import time
import shutil
import socket
from pathlib import Path
import numpy as np
import torch
import datasets
from datasets import load_from_disk
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import Trainer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import evaluate

import src.setup_train
reload(src.setup_train)
from src.setup_train import description
from src.setup_train import path_to_orig_data, path_to_raw_data, path_to_tokenized_data
from src.setup_train import path_to_state_dict, path_to_training_dir
from src.setup_train import device
from src.setup_train import clear_raw_data
from src.setup_train import do_preprocess_orig_data, subset_ratio, do_tokenize_data
from src.setup_train import test_size, valid_size
from src.setup_train import seed
from src.setup_train import ModelClass, checkpoint, num_hidden_layers
from src.setup_train import transformer_head_name, transformer_max_seq_length
from src.setup_train import training_arguments_kw
import src.preprocess_data
reload(src.preprocess_data)
from src.preprocess_data import preprocess_orig_data


##########################################################################
# Preprocess the data to get raw data into /galactica/data
##########################################################################
# Preprocess the data to get raw data into /galactica/data
if do_preprocess_orig_data == True:
    preprocess_orig_data(str(path_to_orig_data), 
                         str(path_to_raw_data), 
                         test_size=test_size,
                         valid_size=valid_size,
                         clear=clear_raw_data)
    
print(f"Processed data available in the file: {path_to_raw_data}")


##########################################################################
# Load the data, model, tokenizer and tokenize
##########################################################################

############################################
if do_tokenize_data:
    # Load the raw dataset
    print('\n' + '*'*50)
    print("Loading the datasets")
    raw_datasets = load_from_disk(str(path_to_raw_data))
    print("\nDatasets loaded: ", raw_datasets)

else:
    # Load the tokenized dataset
    print('\n' + '*'*50)
    print("Loading the datasets")
    raw_datasets = load_from_disk(str(path_to_tokenized_data))
    print("\nDatasets loaded: ", raw_datasets)

############################################
# Get the number of labels
num_labels = len(raw_datasets['train'].features['labels'].names)
print('\n' + '*'*50)
print("Number of labels: ", num_labels)

############################################
# Load the Model
print('\n' + '*'*50)
print("Instantiating the model")
model = ModelClass.from_pretrained(checkpoint, num_labels=num_labels, num_hidden_layers=num_hidden_layers)
if device.type == 'cuda':
    #TODO: understand how to simplify this instantiation
    state_dict = model.state_dict()
    torch.save(state_dict, str(path_to_state_dict))
    config = AutoConfig.from_pretrained(checkpoint, num_labels=num_labels, num_hidden_layers=num_hidden_layers)
    with init_empty_weights():
        model = ModelClass._from_config(config)
    model.tie_weights()
    no_split_module_classes = None #List of modules with any residual connection of some kind
    model = load_checkpoint_and_dispatch(model, str(path_to_state_dict), device_map="auto", no_split_module_classes=no_split_module_classes)
print("\nModel instantiated")

############################################
# Freeze the model but the last layer
print('\n' + '*'*50)
print("Freezing the parameters")
for param in model.named_parameters():
    if param[0] != transformer_head_name:
        param[1].requires_grad = False
print("\nParameters frozen")

############################################
# Load the Tokenizer
print('\n' + '*'*50)
print("Instantiating the tokenizer")
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
id2label = {i: label for label, i in tokenizer.vocab.items()}
pad_token_id = model.config.pad_token_id
tokenizer.add_special_tokens({'pad_token': id2label[pad_token_id]})
tokenizer.model_max_length = transformer_max_seq_length
print("\nTokenizer instantiated")

############################################
if do_tokenize_data:
    # Tokenizing the Dataset
    print('\n' + '*'*50)
    print("Tokenizing the datasets")
    def tokenize_function(sequences):
        return tokenizer(sequences['text'], max_length=transformer_max_seq_length, truncation=True)
    tokenized_datasets = raw_datasets.map(tokenize_function , batched=True)
    print("\nDatasets tokenized: ", tokenized_datasets)

    # Save the datasets to disk
    print('\n' + '*'*50)
    print("Saving the datasets")
    tokenized_datasets.save_to_disk(path_to_tokenized_data)

print(f"\nDatasets available in :{path_to_tokenized_data}")

############################################
try:
    if raw_datasets:
        del(raw_datasets)
except NameError: pass

try:
    if tokenized_datasets:
        del(tokenized_datasets)
except NameError: pass


##########################################################################
# Define the training parameters
##########################################################################

############################################
# Load the dataset
print('\n' + '*'*50)
print("Loading the datasets")
tokenized_datasets = load_from_disk(str(path_to_tokenized_data))
print("\nDatasets loaded: ", tokenized_datasets)

############################################
# Take a subset of tokenized_dataset
print('\n' + '*'*50)
print("Sampling the datasets")
for split in tokenized_datasets:
    subset_size = int(len(tokenized_datasets[split]) * subset_ratio[split])
    tokenized_datasets[split] = tokenized_datasets[split].shuffle(seed=seed).select(range(subset_size))
print("\nDatasets sampled: ", tokenized_datasets)

############################################
# Define the data collator
print('\n' + '*'*50)
print("Instantiating the DataCollator")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
print("\nDataCollator instantiated")

############################################
# Define the TrainingArguments and Trainer
print('\n' + '*'*50)
print("Instantiating TrainingArguments")

current_time = time.strftime("%b%d_%H-%M-%S")
host = socket.gethostname()
logging_dir = Path(path_to_training_dir,'runs',description,current_time + '_' + host)
output_dir = Path(path_to_training_dir,'output_dir',description,current_time + '_' + host)

training_args = TrainingArguments(
    output_dir = str(output_dir),
    logging_dir = str(logging_dir),
    **training_arguments_kw)
print("\n output_dir: ", output_dir)
print("\n logging_dir: ", logging_dir)
print("\nTrainingArguments instantiated")

# Set-up Tensorboard SummaryWriter
print('\n' + '*'*50)
print("Set-up Tensorboard SummaryWriter to enrich training information")

from torch.utils.tensorboard import SummaryWriter
tb_writer = SummaryWriter(logging_dir)

def plot_label_stats(datasets, label_name='labels',  _label_name='_labels'):

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=[6,3], layout='constrained')
    fig.suptitle('Nb of examples per split')

    for i, split in enumerate(datasets):
        dataset = datasets[split]
        labels = dataset.features[label_name].names
        _labels = dataset.features[_label_name].names

        x = range(len(labels))
        height = [dataset[label_name].count(label) for label in labels]
        labs = [label for label in _labels]
        ax[i].bar(x=x, height=height, align='center', label=labs)
        ax[i].set_xlabel(split)
        ax[i].set_xticks(x, labs)
    
    return (fig, ax)
print("\nTensorboard SummaryWriter has been set-up")

############################################
# Cleaning
print('\n' + '*'*50)
print("Cleaning memory")
import gc
gc.collect()

import torch
torch.cuda.empty_cache()
print("\nCleaning done")

############################################
# Define the evaluation computation
print('\n' + '*'*50)
print("Define the evaluation metrics")
def compute_metrics(eval_pred, average = 'weighted'):

    accuracy_metric = evaluate.load('accuracy')
    f1_metric = evaluate.load('f1')
    precision_metric = evaluate.load('precision')
    recall_metric = evaluate.load('recall')
    logits, labels = eval_pred

    # TODO: recheck axis = -
    preds = np.argmax(logits, axis = -1)

    results = {}
    results.update(accuracy_metric.compute(predictions=preds, references=labels))
    results.update(f1_metric.compute(predictions=preds, references=labels, average=average))
    results.update(precision_metric.compute(predictions=preds, references=labels, average=average))
    results.update(recall_metric.compute(predictions=preds, references=labels, average=average))

    return results 
print("\nEvaluation metrics defined")

############################################
# Define the callbacks
print('\n' + '*'*50)
print("Define the  callbacks")
callbacks = None
print("\nCallbacks defined")

############################################
print('\n' + '*'*50)
print("Instantiate the Trainer")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=callbacks,
)

print("\nTrainer instantiated")


##########################################################################
# Train
##########################################################################
# Add the labels stats to Tensorboard
fig, ax = plot_label_stats(tokenized_datasets)
tb_writer.add_figure(tag="labels stats", figure=fig)

# Train the model
output = trainer.train()