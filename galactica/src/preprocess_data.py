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

    # Make sure the labels are strings
    new_features = dataset_dict['train'] .features.copy()
    new_features['labels'] = Value('string')
    dataset_dict = dataset_dict.cast(new_features)

    # Preprocess dataset: split the dataset
    dataset_dict = split_dataset(dataset_dict['train'], test_size, valid_size)

    # Preprocess dataset: encode labels
    # Labels are defined by the train split
    labels = ['[UNK]']
    labels.extend(list(set(dataset_dict['train']['labels'])))

    dataset_dict = dataset_dict.map(lambda seq: {'labels': seq['labels'] if seq['labels'] in labels else '[UNK]'})
    
    # Preprocess dataset: cast to features
    features = Features({
    'id': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
    'title': Value(dtype='string', id=None),
    'text': Value(dtype='string', id=None),
    'labels': ClassLabel(num_classes=len(labels), names=labels)
    })
    dataset_dict = dataset_dict.cast(features)

    # Save the dataset
    dataset_dict.save_to_disk(raw_path)
