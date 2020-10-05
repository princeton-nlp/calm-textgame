import torch
import os
import json
import glob
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


def get_dataset(exclude, data_directory, tokenizer, max_len=-1, shuffle_trajectories=False, data_percentage=1):
    token_id_set, act_mask_set = [], []
    files = []
    for filename in glob.glob(os.path.join(data_directory, '*')):
        if os.path.basename(filename) in exclude:
            continue
        else:
            files.append(filename)

    if data_percentage != 1:
        files = np.random.choice(files, size=int(data_percentage * len(files)), replace=False)

    if shuffle_trajectories:
        np.random.shuffle(files)

    for filename in files:
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                token_ids, act_mask = process(line, tokenizer)
                if max_len == -1 or len(token_ids) < max_len:
                    token_id_set.append(token_ids)
                    act_mask_set.append(act_mask)
    return token_id_set, act_mask_set


def process(line, tokenizer):
    """
    Process each line of the dataset to tokens and action masks.
    :param act_len: Pad or cut action length to act_len. 7 for BERT model, 1 for verb model, -1 for doing nothing.
    if -2 this means we mask out the last sep token for gpt-2
    """
    # Turn [STATE] and [ACTION] to [SEP]
    words = line.split()
    while "" in words:
        words.remove("")
    words = ["[SEP]" if w in ["[STATE]", "[ACTION]"] else w for w in words]
    words[0] = "[CLS]"
    line = " ".join(words)

    # Find where the last action starts, and cut or pad when needed
    tokens = tokenizer.tokenize(line, add_prefix_space=True)
    act_pos = len(tokens) - tokens[::-1].index("[SEP]")
    tokens += ["[SEP]"]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Act mask
    act_mask = np.zeros(len(tokens))
    act_mask[act_pos:] = 1

    return token_ids, act_mask

def pad_sequences(data, pad_length, dtype):
    padded_data = np.zeros((len(data), pad_length), dtype=dtype)
    for i, line in enumerate(data):
        if len(line) > pad_length:
            line = line[len(line) - pad_length:]
        padded_data[i,:len(line)] = line
    return padded_data

def train_test_split(data, validate_size=0.9):
    token_ids, act_masks, att_masks = data
    indices = [i for i in range(len(token_ids))]
    train_idx = indices[:int(validate_size * len(indices))]
    validate_idx = indices[int(validate_size * len(indices)):]

    train_inputs = token_ids[train_idx]
    val_inputs = token_ids[validate_idx]
    train_act_masks = act_masks[train_idx]
    val_act_masks = act_masks[validate_idx]
    train_att_masks = att_masks[train_idx]
    val_att_masks = att_masks[validate_idx]

    return train_inputs, val_inputs, train_act_masks, val_act_masks, train_att_masks, val_att_masks


def get_dataloader(exclude, data_directory, tokenizer, max_len=256, bs=16, shuffle_trajectories=False,
                   data_percentage=1):
    n_gpu = torch.cuda.device_count()
    per_gpu_batch_size = bs
    batch_size = max(1, n_gpu) * per_gpu_batch_size
    print("Number of GPU: " + str(n_gpu))

    token_id_set, act_mask_set = get_dataset(exclude, data_directory, tokenizer, max_len=max_len, \
                                             shuffle_trajectories=shuffle_trajectories, data_percentage=data_percentage)
    att_mask_set = [np.ones(len(ids)) for ids in token_id_set]
    print(str(len(token_id_set)) + " examples in dataset")
    print("Data Sample\n", tokenizer.convert_ids_to_tokens(token_id_set[0]), '\n', act_mask_set[0])

    token_ids = pad_sequences(token_id_set, 256, 'int')
    act_masks = pad_sequences(act_mask_set, 256, 'uint8')
    att_masks = pad_sequences(att_mask_set, 256, 'uint8')

    train_inputs, val_inputs, train_act_masks, val_act_masks, train_att_masks, val_att_masks \
        = train_test_split((token_ids, act_masks, att_masks), validate_size=0.9)

    train_data = TensorDataset(torch.tensor(train_inputs), torch.tensor(train_att_masks), torch.tensor(train_act_masks))
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size,
                                  drop_last=True)  # drop last batch for gpt-2

    val_data = TensorDataset(torch.tensor(val_inputs), torch.tensor(val_att_masks), torch.tensor(val_act_masks))
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=1)
    return train_dataloader, val_dataloader
