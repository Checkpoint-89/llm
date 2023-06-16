from pathlib import Path
import shutil
import numpy as np
from datasets import load_dataset, DatasetDict
from datasets import Features, Value, Sequence, ClassLabel
from .setup_train import norm_col_names, raw_data_col_names

def split_dataset(dataset, test_size, valid_size=None):

    # Get a Dataset and return a Datasetdict with its splits"

    assert test_size + valid_size < 1, "test_size + valid_size should be less then 1."

    ds_dict = dataset.train_test_split(test_size=test_size + valid_size)
    train = ds_dict['train']

    if valid_size == None:
        return ds_dict

    test_size = valid_size / (test_size + valid_size)
    dataset = ds_dict['test']
    ds_dict = dataset.train_test_split(test_size=test_size)

    test = ds_dict['train']
    valid = ds_dict['test']

    return DatasetDict({'train': train, 'test': test, 'validation': valid})


def rename_ds_columns(dataset, src_cols, tgt_cols):

    for src, tgt in zip(src_cols, tgt_cols):
        try:
            dataset = dataset.rename_column(src, tgt)
        except ValueError as e:
            pass
            # print("Pass, ", e)
    return dataset


def preprocess_orig_data(
        orig_path,
        raw_path,
        test_size=0.1,
        valid_size=0.1,
        raw_data_col_names=raw_data_col_names,
        norm_col_names=norm_col_names,
        clear=False):
    """
    Preprocess the original data provided by Chiper Icon
    Inputs
    - /galactica/data/orig_applications.json or /galactica/data/orig_is_experimental.json
    Outputs
    - /galactica/data/raw_applications.json or /galactica/data/raw_is_experimental.json
    """
    #TODO: Add distribution statistics as metadata to the dataset

    # Clear the directory at raw_path
    if clear:
        shutil.rmtree(raw_path, ignore_errors=True)

    # Load the data
    # Note: HF will
    #     - load the data
    #     - wrap it in a DatasetDict (all data in 'train')
    #     - store is as an Apache Arrow file
    dataset_dict = load_dataset('json', data_files=str(orig_path)) 

    # Preprocess the dataset
    tgt_cols = norm_col_names

    # Preprocess dataset: standardize column names
    src_cols = list(dataset_dict['train'].features.keys())
    assert src_cols == raw_data_col_names, "column names or order have changed !"

    dataset_dict['train'] = rename_ds_columns(dataset_dict['train'], src_cols, tgt_cols)

    # Preprocess dataset: split the dataset
    dataset_dict = split_dataset(dataset_dict['train'], test_size, valid_size)

    # Preprocess dataset: encode labels
    # Labels are defined by the train split
    labels = list(set(dataset_dict['train']['_labels']))
    label2id = {label: i for i, label in enumerate(labels)}
    label2id['[UNK]'] = -1
    id2label = {i: label for i, label in enumerate(labels)}
    id2label[-1] = '[UNK]'

    dataset_dict = dataset_dict.map(lambda seq: {'labels': label2id[seq['_labels']] if seq['_labels'] in label2id.keys() else -1})
    for split in dataset_dict.keys():
        if split != 'train':
            dataset_dict[split] = dataset_dict[split].map(lambda seq: {'_labels': id2label[seq['labels']]})

    # Preprocess dataset: cast to features
    features = Features({
    '_labels': ClassLabel(num_classes=len(label2id.keys()), names=list(label2id.keys())),
    'id': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
    'title': Value(dtype='string', id=None),
    'text': Value(dtype='string', id=None),
    'labels': ClassLabel(num_classes=len(label2id.values()), names=list(label2id.values())),
    })
    dataset_dict = dataset_dict.cast(features)

    # Save the dataset
    dataset_dict.save_to_disk(raw_path)

    # Save the dataset
    dataset_dict.save_to_disk(raw_path)
