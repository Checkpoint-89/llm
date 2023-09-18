import os
import shutil
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, OPTForSequenceClassification, OPTForCausalLM, OPTModel

##########################
# my specific setup
##########################

# what to do
make_tensors = True # Compute embeddings vectors
make_metadata = True # Compute metadata

# paths to the root folder of the project
# my_path_to_galactica_folder = Path(r'/home/cedric.dietzi/projects/galactica') # path to root folder
my_path_to_galactica_folder = Path(r'D:\Users\cdiet\Documents\projects\llm\galactica') # path to root folder

# data to process
my_data = 'applications'                # on what to work: 'is_experimental' or 'applications'

model_type = r'openai/text-embedding-ada-002'
checkpoint_model = r'text-embedding-ada-002'

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

# path to the embeddings
output_dir = Path(r'./test-trainer/embds/', model_type, data_type)


##########################
# programmatic setup
##########################

# set up the path to the original data based on data_type
if data_type == 'applications':
    path_to_orig_data = Path(path_to_galactica_folder, orig_applications)
    path_to_raw_data = Path(path_to_galactica_folder, raw_applications)
elif data_type == 'is_experimental':
    path_to_orig_data = Path(path_to_galactica_folder, orig_is_experimental)
    path_to_raw_data = Path(path_to_galactica_folder, raw_is_experimental)
else:
    raise ValueError("Unknown data type !")