import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
import json
import numpy as np
from train_config.globalpointer_config import Config
from utils.tools import token_rematch


def load_data(filename):
    resultList = []
    """加载数据
    单条格式：[text, (start, end, label), (start, end, label), ...]，
              意味着text[start:end + 1]是类型为label的实体。
    """
    D = []
    for d in json.load(open(filename)):
        D.append([d['text']])
        for e in d['entities']:
            start, end, label, entity = e['start_idx'], e['end_idx'], e['type'], e['entity']
            if start <= end:
                D[-1].append((start, end, label, entity))
            resultList.append(label)
    categories = list(set(resultList))
    categories.sort(key=resultList.index)
    return D


class NerDataset(Dataset):
    def __init__(self, data, tokenizer, label2id, max_len, model_type='span'):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id
        self.model_type = model_type  # ['softmax', 'crf', 'span', 'span_v2', 'global_pointer']
        self.result = []

        for d in self.data:
            if self.model_type == 'global_pointer':
                text = d[0]
                input_id = tokenizer.encode(text)
                if len(input_id) > self.max_len:
                    input_id = [101] + input_id[1:self.max_len - 1] + [102]

                input_mask = [1] * len(input_id)

                label = np.zeros((len(self.label2id), len(input_id), len(input_id)))
                tokens = tokenizer.tokenize(d[0], max_length=self.max_len, add_special_tokens=True,
                                            truncation=True)
                mapping = token_rematch().rematch(d[0], tokens)
                start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
                end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
                for entity_input in d[1:]:
                    start, end = entity_input[0], entity_input[1]
                    if start in start_mapping and end in end_mapping and start < self.max_len and end < self.max_len:
                        start = start_mapping[start]
                        end = end_mapping[end]
                        label[self.label2id[entity_input[2]], start, end] = 1
                self.result.append((input_id, input_mask, label, model_type))

            elif model_type == 'crf' or model_type == 'softmax':
                text = d[0]
                words = list(text)
                if len(words) > self.max_len - 2:
                    words = ['[CLS]'] + words[1:self.max_len - 1] + ['[SEP]']
                else:
                    words = ['[CLS]'] + words + ['[SEP]']
                input_id = tokenizer.convert_tokens_to_ids(words)
                label = ['O'] * (len(input_id) - 2)
                input_mask = [1] * len(input_id)

                for entity_input in d[1:]:
                    start, end, label_type, entity = entity_input[0], entity_input[1], entity_input[2], \
                                                     entity_input[3]
                    assert ''.join(text[start: end + 1]) == entity
                    if start >= len(label) or end >= len(label):
                        continue  # 被截断
                    if start == end:
                        label[start] = 'S-' + label_type
                    else:
                        label[start] = 'B-' + label_type
                        label[start + 1: end + 1] = ['I-' + label_type] * (len(entity) - 1)

                label = ['O'] + label + ['O']
                label = [self.label2id[x] for x in label]  # label -> id

                self.result.append((input_id, input_mask, label, model_type))

            elif model_type == 'span':
                text = d[0]
                words = list(text)
                if len(words) > self.max_len - 2:
                    words = ['[CLS]'] + words[1:self.max_len - 1] + ['[SEP]']
                else:
                    words = ['[CLS]'] + words + ['[SEP]']
                input_id = tokenizer.convert_tokens_to_ids(words)
                input_mask = [1] * len(input_id)

                start_ids = [0] * (len(input_id) - 2)  # 10在span_id2label中表示'o'
                end_ids = [0] * (len(input_id) - 2)
                subject_id = []
                for entity_input in d[1:]:
                    start, end, label_type, entity = entity_input[0], entity_input[1], entity_input[2], \
                                                     entity_input[3]
                    assert ''.join(text[start: end + 1]) == entity
                    if start >= len(start_ids) or end >= len(start_ids):
                        continue  # 被截断
                    # span
                    start_ids[start] = self.label2id[label_type]
                    end_ids[end] = self.label2id[label_type]
                    subject_id.append((self.label2id[label_type], start, end))

                # span
                start_ids = [self.label2id['o']] + start_ids + [self.label2id['o']]
                end_ids = [self.label2id['o']] + end_ids + [self.label2id['o']]

                self.result.append((input_id, input_mask, start_ids, end_ids, subject_id, model_type))

            else:  # span_v2
                text = d[0]
                words = list(text)
                if len(words) > self.max_len - 2:
                    words = ['[CLS]'] + words[1:self.max_len - 1] + ['[SEP]']
                else:
                    words = ['[CLS]'] + words + ['[SEP]']
                input_id = tokenizer.convert_tokens_to_ids(words)
                input_mask = [1] * len(input_id)

                start_v2_ids = np.zeros((len(self.label2id), len(input_id)))
                end_v2_ids = np.zeros((len(self.label2id), len(input_id)))
                subject_id = []

                for entity_input in d[1:]:
                    start, end, label_type, entity = entity_input[0], entity_input[1], entity_input[2], \
                                                     entity_input[3]
                    assert ''.join(text[start: end + 1]) == entity
                    if start >= len(input_id) or end >= len(input_id):
                        continue  # 被截断
                    start_v2_ids[self.label2id[label_type], start] = 1
                    end_v2_ids[self.label2id[label_type], end] = 1
                    subject_id.append((self.label2id[label_type], start, end))

                self.result.append((input_id, input_mask, start_v2_ids, end_v2_ids, subject_id, model_type))

    def __len__(self):
        return len(self.result)

    def __getitem__(self, idx):
        output = self.result[idx]
        if output[-1] == 'global_pointer' or output[-1] == 'crf' or output[-1] == 'softmax':
            return output[0], output[1], output[2]

        elif output[-1] == 'span' or output[-1] == 'span_v2':
            return output[0], output[1], output[2], output[3], output[4]


def yeild_data(file_data, label2id, model_type, is_train, ddp=True):
    config = Config()
    tokenizer = BertTokenizerFast.from_pretrained(config.pretrained_model_path, do_lower_case=True)
    if is_train:
        train_data = load_data(file_data)
        train_data = NerDataset(train_data, tokenizer, label2id, config.max_len, model_type)
        if ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
            if model_type == 'span':
                train_dataloader = DataLoader(train_data, batch_size=config.batch_size, sampler=train_sampler,
                                              shuffle=False, collate_fn=collate_to_max_length_span)
            elif model_type == 'span_v2':
                train_dataloader = DataLoader(train_data, batch_size=config.batch_size, sampler=train_sampler,
                                              shuffle=False, collate_fn=collate_to_max_length_span_v2)
            else:
                train_dataloader = DataLoader(train_data, batch_size=config.batch_size, sampler=train_sampler,
                                              shuffle=False, collate_fn=collate_to_max_length)
        else:
            if model_type == 'span':
                train_dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True,
                                              collate_fn=collate_to_max_length_span)
            elif model_type == 'span_v2':
                train_dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True,
                                              collate_fn=collate_to_max_length_span_v2)
            else:
                train_dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True,
                                              collate_fn=collate_to_max_length)
        return train_dataloader
    else:
        dev_data = load_data(file_data)
        dev_data = NerDataset(dev_data, tokenizer, label2id, config.max_len, model_type)
        if ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dev_data)
            if model_type == 'span':
                dev_dataloader = DataLoader(dev_data, batch_size=config.batch_size, sampler=train_sampler,
                                            collate_fn=collate_to_max_length_span)
            elif model_type == 'span_v2':
                dev_dataloader = DataLoader(dev_data, batch_size=config.batch_size, sampler=train_sampler,
                                            collate_fn=collate_to_max_length_span_v2)
            else:  # global_pointer, crf, softmax
                dev_dataloader = DataLoader(dev_data, batch_size=config.batch_size, sampler=train_sampler,
                                            collate_fn=collate_to_max_length)
        else:
            if model_type == 'span':
                dev_dataloader = DataLoader(dev_data, batch_size=config.batch_size,
                                            collate_fn=collate_to_max_length_span)
            elif model_type == 'span_v2':
                dev_dataloader = DataLoader(dev_data, batch_size=config.batch_size,
                                            collate_fn=collate_to_max_length_span_v2)
            else:
                dev_dataloader = DataLoader(dev_data, batch_size=config.batch_size,
                                            collate_fn=collate_to_max_length)
        return dev_dataloader


def collate_to_max_length(batch):
    #  input_id, input_mask, label
    batch_size = len(batch)
    input_id_list = []
    input_mask_list = []
    label_list = []

    for single_data in batch:
        input_id_list.append(single_data[0])
        input_mask_list.append(single_data[1])
        label_list.append(single_data[2])

    max_length = max([len(item) for item in input_id_list])
    num_labels = len(label_list[0])

    output = [torch.full([batch_size, max_length],
                         fill_value=0,
                         dtype=torch.long),
              torch.full([batch_size, max_length],
                         fill_value=0,
                         dtype=torch.long)
              ]
    for i in range(batch_size):
        output[0][i][0:len(input_id_list[i])] = torch.LongTensor(input_id_list[i])
        output[1][i][0:len(input_mask_list[i])] = torch.LongTensor(input_mask_list[i])

    if isinstance(label_list[0], list):
        output.append(torch.full([batch_size, max_length], fill_value=0, dtype=torch.long))
        for i in range(batch_size):
            output[2][i][0:len(label_list[i])] = torch.LongTensor(label_list[i])
    else:
        output.append(torch.full([batch_size, num_labels, max_length, max_length], fill_value=0, dtype=torch.long))
        for i in range(batch_size):
            output[2][i, :, 0:len(label_list[i][0]), 0:len(label_list[i][0])] = torch.LongTensor(label_list[i])

    return output  # input_id, input_mask, label


def collate_to_max_length_span(batch):
    #  input_id, input_mask, start_ids, end_ids, subject_id
    batch_size = len(batch)
    input_id_list = []
    input_mask_list = []
    start_ids_list = []
    end_ids_list = []
    subject_list = []

    for single_data in batch:
        input_id_list.append(single_data[0])
        input_mask_list.append(single_data[1])
        start_ids_list.append(single_data[2])
        end_ids_list.append(single_data[3])
        subject_list.append(single_data[4])

    max_length = max([len(item) for item in input_id_list])

    output = [torch.full([batch_size, max_length],
                         fill_value=0,
                         dtype=torch.long),
              torch.full([batch_size, max_length],
                         fill_value=0,
                         dtype=torch.long),
              torch.full([batch_size, max_length],
                         fill_value=0,
                         dtype=torch.long),
              torch.full([batch_size, max_length],
                         fill_value=0,
                         dtype=torch.long)
              ]
    for i in range(batch_size):
        output[0][i][0:len(input_id_list[i])] = torch.LongTensor(input_id_list[i])
        output[1][i][0:len(input_mask_list[i])] = torch.LongTensor(input_mask_list[i])
        output[2][i][0:len(start_ids_list[i])] = torch.LongTensor(start_ids_list[i])
        output[3][i][0:len(end_ids_list[i])] = torch.LongTensor(end_ids_list[i])
    output.append(subject_list)
    return output  # input_id, input_mask, start_ids, end_ids, subject


def collate_to_max_length_span_v2(batch):
    #  input_id, input_mask, start_ids, end_ids, subject_id
    batch_size = len(batch)
    input_id_list = []
    input_mask_list = []
    start_v2_ids_list = []
    end_v2_ids_list = []
    subject_list = []

    for single_data in batch:
        input_id_list.append(single_data[0])
        input_mask_list.append(single_data[1])
        start_v2_ids_list.append(single_data[2])
        end_v2_ids_list.append(single_data[3])
        subject_list.append(single_data[4])

    max_length = max([len(item) for item in input_id_list])
    num_labels = len(start_v2_ids_list[0])

    output = [torch.full([batch_size, max_length],
                         fill_value=0,
                         dtype=torch.long),
              torch.full([batch_size, max_length],
                         fill_value=0,
                         dtype=torch.long),
              torch.full([batch_size, num_labels, max_length],
                         fill_value=0,
                         dtype=torch.long),
              torch.full([batch_size, num_labels, max_length],
                         fill_value=0,
                         dtype=torch.long)
              ]
    for i in range(batch_size):
        output[0][i][0:len(input_id_list[i])] = torch.LongTensor(input_id_list[i])
        output[1][i][0:len(input_mask_list[i])] = torch.LongTensor(input_mask_list[i])
        output[2][i, :, 0:len(start_v2_ids_list[i][0])] = torch.LongTensor(start_v2_ids_list[i])  # [num_labels, max_length]
        output[3][i, :, 0:len(end_v2_ids_list[i][0])] = torch.LongTensor(end_v2_ids_list[i])
    output.append(subject_list)
    return output  # input_id, input_mask, start_v2_ids_list, end_v2_ids_list, subject
