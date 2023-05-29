from pathlib import Path
import shutil
import numpy as np
from datasets import load_dataset, DatasetDict
from datasets import Features, Value, Sequence, ClassLabel

def split_dataset(dataset, test_size, valid_size=None):

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


def preprocess_orig_data(path_to_galactica_folder ,path_to_ds, clean=False):
    """
    Preprocess the original data provided by Chiper Icon
    Inputs, paths to
    - /galactica/data/orig_applications.json or /galactica/data/orig_is_experimental.json
    Outputs
    - /galactica/data/raw_applications.json or /galactica/data/raw_is_experimental.json
    """

    # Identify the data
    # TODO: remove all hard coded paths
    if Path(path_to_ds) == Path(r'./data/orig_applications.json'):
        data_type = 'applications'
    elif Path(path_to_ds) == Path(r'./data/orig_is_experimental.json'):
        data_type = 'is_experimental'
    else:
        raise ValueError("Unknown data type !")
    
    # Set the path to save the dataset
    if data_type == 'applications':
        raw_path = Path(path_to_galactica_folder, r'./data/raw_applications.json')
    elif data_type == 'is_experimental':
        raw_path = Path(path_to_galactica_folder, r'./data/raw_is_experimental.json')

    # Cleaning
    if clean:
        shutil.rmtree(raw_path, ignore_errors=True)
    
    # Load the data
    orig_path = Path(path_to_galactica_folder, path_to_ds)
    dataset = load_dataset('json', data_files=str(orig_path))

    # Preprocess the dataset
    tgt_cols = ['_labels', 'id', 'title', 'text']

    # Preprocess dataset: standardize column names
    src_cols = list(dataset['train'].features.keys())
    if data_type == 'applications':
        assert src_cols == ['application', 'ids', 'section_title', 'section_text'], "column names or order have changed !"
    elif data_type == 'is_experimental':
        assert src_cols == ['is_experimental_section', 'ids', 'section_title', 'section_text'], "column names or order have changed !"
    else:        
        raise ValueError("Unknown data type !")
    dataset['train'] = rename_ds_columns(dataset['train'], src_cols, tgt_cols)

    # Preprocess dataset: split the dataset
    test_size = 0.1
    valid_size = 0.1
    dataset = split_dataset(dataset['train'], test_size, valid_size)

    # Preprocess dataset: encode labels
    # Labels are defined by the train split
    labels = np.unique(dataset['train']['_labels'])
    label2id = {label: i for i, label in enumerate(labels)}
    label2id['[UNK]'] = -1
    id2label = {i: label for i, label in enumerate(labels)}
    id2label[-1] = '[UNK]'

    dataset = dataset.map(lambda seq: {'labels': label2id[seq['_labels']] if seq['_labels'] in label2id.keys() else -1})
    for split in dataset.keys():
        if split != 'train':
            dataset[split] = dataset = dataset.map(lambda seq: {'_labels': id2label[seq['labels']]})

    #TODO: Add distribution statistics as metadata to the dataset

    # Preprocess dataset: cast to features
    features = Features({
    '_labels': ClassLabel(num_classes=len(label2id.keys()), names=list(label2id.keys())),
    'id': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
    'title': Value(dtype='string', id=None),
    'text': Value(dtype='string', id=None),
    'labels': ClassLabel(num_classes=len(label2id.values()), names=list(label2id.values())),
    })
    dataset = dataset.cast(features)

    # Save the dataset
    dataset.save_to_disk(raw_path)