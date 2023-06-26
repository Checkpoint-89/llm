import os
import shutil
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, OPTForSequenceClassification

##########################
# my specific setup
##########################

# paths to the root folder of the project
my_path_to_galactica_folder = Path(r'/home/cedric.dietzi/projects/galactica') # path to root folder
# my_path_to_galactica_folder = Path(r'D:\Users\cdiet\Documents\projects\llm\galactica') # path to root folder

# whether to clear directories
clear_state_dict_dir = False            # whether to clear the directory of the state_dict
clear_tensor_board_directory = False    # whether to clear the tensorboard directory

# data to process
my_data = 'is_experimental'                # on what to work: 'is_experimental' or 'applications'

# preprocessing
do_preprocess_orig_data = False         # whether to preprocess the original data into raw data
clear_raw_data = True                  # whether to clear existing data when repreprocessing

# tokenizing
do_tokenize_data = True                # whether to tokenize the raw data into tokenized data
# transformer_max_seq_length = 1024       # This has been replaced with a programmatic setup in the code
                                        # => test showed 4096 is okay for the tokenizer, but may be the reason for the training top fail
                                        # TODO: investigate the impact of this parameter on the the training of OPTForSequenceClassification

if my_data == 'applications':           # what ratio of the raw data to sample for the various intented treatment
    subset_ratio = {'train': 0.30, 'test': 0.10, 'validation': 0.10}
elif my_data == 'is_experimental':
    subset_ratio = {'train': 0.15, 'test': 0.05, 'validation': 0.05}
    

# model
ModelClass = AutoModelForSequenceClassification
checkpoint = "allenai/scibert_scivocab_uncased"

# ModelClass = OPTForSequenceClassification # https://huggingface.co/docs/transformers/model_doc/opt#opt
# checkpoint = "facebook/galactica-125m"

# transformer_head_name = 'score.weight'    # This has been replaced with a programmatic setup in the code
num_hidden_layers = 12                          # 12

# training arguments
my_seq_per_batch = 2
# first value on the comment line is the default value
training_arguments_kw = dict(
run_name = 'test_run',                   # A descriptor for the run. Typically used for wandb and mlflow logging.

#output_dir,                             # None
overwrite_output_dir = False,            # False

num_train_epochs = 1,                    # 3
per_device_train_batch_size = my_seq_per_batch,        # 8 /The batch size per GPU/TPU core/CPU for training.
per_device_eval_batch_size = my_seq_per_batch,         # 8 /The batch size per GPU/TPU core/CPU for evaluation
gradient_accumulation_steps = 1,         # 1 /Number of updates steps to accumulate the gradients for, before performing a backward/update pass.

learning_rate = 5e-5,                    # 5e-5
lr_scheduler_type = 'linear',            # "linear" /The scheduler type to use. See the documentation of SchedulerType.
weight_decay = 0,                        # 0
warmup_steps = 0,                        # 0

log_level = 'warning',                   # 'passive'~'warning', 'debug', 'info', 'warning', 'error' and 'critical' /Logger log level to use on the main process.
log_level_replica = 'warning',           # 'passive'~'warning', 'debug', 'info', 'warning', 'error' and 'critical' /Logger log level to use on replicas.
log_on_each_node = True,                 # True /In multinode distributed training, whether to log using log_level once per node, or only on the main node.
logging_strategy = 'steps',              # 'steps', 'no', 'epoch /The logging strategy to adopt during training.
logging_steps = int(2048/my_seq_per_batch), # 500 /Number of update steps between two logs if logging_strategy="steps". Should be an integer or a float in range [0,1).
logging_first_step = False,              # False /Whether to log and evaluate the first global_step or not.
logging_nan_inf_filter = True,           # True /Whether to filter out nan and inf losses for logging (replaced by average loss of the current window).
#logging_dir,                            # output_dir/runs/CURRENT_DATETIME_HOSTNAME /Tensorboard log directory.

evaluation_strategy='steps',             # 'no', 'epoch', 'steps' /The evaluation strategy to adopt during training.
eval_steps=int(2048/my_seq_per_batch),    # logging_steps /Number of update steps between two evaluations if evaluation_strategy="steps". Should be an integer or a float in range [0,1)

save_strategy='steps',                   # 'steps', 'no', 'epoch' /The checkpoint save strategy to adopt during training.
save_steps=int(2048/my_seq_per_batch),    # 500 /Number of updates steps before two checkpoint saves if save_strategy="steps". Should be an integer or a float in range [0,1).
save_total_limit = 10,                   # None /If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in output_dir.

load_best_model_at_end = True,           # False /Whether or not to load the best model found during training at the end of training.
metric_for_best_model = 'loss',          # 'loss' /Must be the name of a metric returned by the evaluation with or without the prefix "eval_".
#greater_is_better,                      # True if metric_for_best_model != 'loss' or 'eval_loss' else False /Specifies if better models should have a greater metric or not

report_to = "tensorboard",               # 'all', 'none', 'azure_ml', 'comet_ml', 'mlflow', 'neptune', 'tensorboard','clearml', 'wandb' 
                                         # /The list of integrations to report the results and logs to.
)

##########################
# data
##########################
data_type = my_data

##########################
# deep learning
##########################
dl_framework = 'pt'

seed = None

is_gpu = torch.cuda.is_available()
if is_gpu:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

##########################
# paths
##########################

# path to root folder
path_to_galactica_folder = my_path_to_galactica_folder

# path from root to original data
orig_applications = Path(r'./data/orig_applications.json')
orig_is_experimental = Path(r'./data/orig_is_experimental.json')

# path to raw data (preprocessed into a Dataset Dict)
raw_applications = Path(r'./data/raw_applications.json')
raw_is_experimental = Path(r'./data/raw_is_experimental.json')

# path to the tokenized datasets
if checkpoint == "facebook/galactica-125m":
    tokenized_applications = Path(r'./data/galactica-125m/tokenized_applications.json')
    tokenized_is_experimental = Path(r'./data/galactica-125m/tokenized_is_experimental.json')
elif checkpoint == "allenai/scibert_scivocab_uncased":
    tokenized_applications = Path(r'./data/scibert_scivocab_uncased/tokenized_applications.json')
    tokenized_is_experimental = Path(r'./data/scibert_scivocab_uncased/tokenized_is_experimental.json')
else:
    raise ValueError("checkpoint not recognized")

# path to the model state_dict
path_to_state_dict = Path(r"./state_dict/model_state_dict.pt")
path_to_state_dict = Path(path_to_galactica_folder, path_to_state_dict)

# path to the training directory and tensorboard
path_to_training_dir = Path(r"./test-trainer")
path_to_training_dir = Path(path_to_galactica_folder, path_to_training_dir)
path_to_tensorboard_dir = Path(path_to_training_dir, 'runs')


##########################
# preprocessing
##########################

# column names of the original data and normalized form
col_names_applications = ['application', 'ids', 'section_title', 'section_text']
col_names_is_experimental = ['is_experimental_section', 'ids', 'section_title', 'section_text']
norm_col_names = ['labels', 'id', 'title', 'text']

# train test split
test_size = 0.1
valid_size = 0.1

##########################
# experiment description
##########################
description = str(Path(checkpoint, data_type))

##########################
# programmatic setup
##########################

# set up the path to the original data based on data_type
if data_type == 'applications':
    path_to_orig_data = Path(path_to_galactica_folder, orig_applications)
    path_to_raw_data = Path(path_to_galactica_folder, raw_applications)
    path_to_tokenized_data = Path(path_to_galactica_folder, tokenized_applications)
    raw_data_col_names = col_names_applications
elif data_type == 'is_experimental':
    path_to_orig_data = Path(path_to_galactica_folder, orig_is_experimental)
    path_to_raw_data = Path(path_to_galactica_folder, raw_is_experimental)
    path_to_tokenized_data = Path(path_to_galactica_folder, tokenized_is_experimental)
    raw_data_col_names = col_names_is_experimental
else:
    raise ValueError("Unknown data type !")

# clear the state_dict directory
if clear_state_dict_dir:
    path_to_state_dict_dir = path_to_state_dict.parent
    shutil.rmtree(path_to_state_dict_dir, ignore_errors=True)
    os.mkdir(path_to_state_dict_dir)

# clear the tensorboard directory
if clear_tensor_board_directory:
    shutil.rmtree(path_to_tensorboard_dir, ignore_errors=True)