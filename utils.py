import os
from collections import defaultdict
from tqdm import tqdm
import torch
import numpy as np


class RoSTERUtils(object):

    def __init__(self, data_dir, tokenizer):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.read_types(self.data_dir)

    def read_file(self, file_dir="conll", dataset_name="train", supervision="dist"):
        text_file = os.path.join(file_dir, f"{dataset_name}_text.txt")
        f_text = open(text_file)
        text_contents = f_text.readlines()
        label_file = os.path.join(file_dir, f"{dataset_name}_label_{supervision}.txt")
        f_label = open(label_file)
        label_contents = f_label.readlines()
        sentences = []
        labels = []
        for text_line, label_line in zip(text_contents, label_contents):
            sentence = text_line.strip().split()
            label = label_line.strip().split()
            assert len(sentence) == len(label)
            sentences.append(sentence)
            labels.append(label)
        return sentences, labels

    def read_dict(self, dict_dir):
        _, _, dict_files = next(os.walk(dict_dir))
        self.entity_dict = defaultdict(list)
        self.entity_types = []
        for dict_file in dict_files:
            contents = open(os.path.join(dict_dir, dict_file)).readlines()
            entities = [content.strip() for content in contents]
            entity_type = dict_file.split('.')[0]
            self.entity_types.append(entity_type)
            for entity in entities:
                self.entity_dict[entity].append(entity_type)
            print(f"{entity_type} type has {len(entities)} entities")

        for entity in self.entity_dict:
            if len(self.entity_dict[entity]) > 1:
                print(self.entity_dict[entity])
                exit()

    def read_types(self, file_path):
        type_file = open(os.path.join(file_path, 'types.txt'))
        types = [line.strip() for line in type_file.readlines()]
        self.entity_types = []
        for entity_type in types:
            if entity_type != "O":
                self.entity_types.append(entity_type.split('-')[-1])

    def get_label_map(self, tag_scheme):
        label_map = {'O': 0}
        num_labels = 1
        for entity_type in self.entity_types:
            label_map['B-'+entity_type] = num_labels
            if tag_scheme == 'iob':
                label_map['I-'+entity_type] = num_labels + 1
                num_labels += 2
            elif tag_scheme == 'io':
                label_map['I-'+entity_type] = num_labels
                num_labels += 1
        label_map['UNK'] = -100
        inv_label_map = {k: v for v, k in label_map.items()}
        self.label_map = label_map
        self.inv_label_map = inv_label_map
        return label_map, inv_label_map

    def get_data(self, dataset_name, supervision='true'):
        sentences, labels = self.read_file(self.data_dir, dataset_name, supervision)
        sent_len = [len(sent) for sent in sentences]
        print(f"****** {dataset_name} set stats (before tokenization): sentence length: {np.average(sent_len)} (avg) / {np.max(sent_len)} (max) ******")
        data = []
        for sentence, label in zip(sentences, labels):
            text = ' '.join(sentence)
            label = label
            data.append((text, label))
        return data

    def get_tensor(self, dataset_name, max_seq_length, supervision="true", drop_o_ratio=0):
        data_file = os.path.join(self.data_dir, f"{dataset_name}_{supervision}.pt")
        if os.path.exists(data_file):
            print(f"Loading data from {data_file}")
            tensor_data = torch.load(data_file)
        else:
            all_data = self.get_data(dataset_name=dataset_name, supervision=supervision)
            raw_labels = [data[1] for data in all_data]
            all_input_ids = []
            all_attention_mask = []
            all_labels = []
            all_valid_pos = []
            for text, labels in tqdm(all_data, desc="Converting to tensors"):
                encoded_dict = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=max_seq_length, 
                                                          padding='max_length', return_attention_mask=True, truncation=True, return_tensors='pt')
                input_ids = encoded_dict['input_ids']
                attention_mask = encoded_dict['attention_mask']
                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
                label_idx = -100 * torch.ones(max_seq_length, dtype=torch.long)
                valid_pos = torch.zeros(max_seq_length, dtype=torch.long)
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
                j = 0
                for i, token in enumerate(tokens[1:], start=1):  # skip [CLS]
                    if token == self.tokenizer.sep_token:
                        break
                    if i == 1 or token.startswith('Ġ'):
                        label = labels[j]
                        label_idx[i] = self.label_map[label]
                        valid_pos[i] = 1
                        j += 1
                assert j == len(labels) or i == max_seq_length - 1
                all_labels.append(label_idx.unsqueeze(0))
                all_valid_pos.append(valid_pos.unsqueeze(0))
                
            all_input_ids = torch.cat(all_input_ids, dim=0)
            all_attention_mask = torch.cat(all_attention_mask, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            all_valid_pos = torch.cat(all_valid_pos, dim=0)
            all_idx = torch.arange(all_input_ids.size(0))
            tensor_data = {"all_idx": all_idx, "all_input_ids": all_input_ids, "all_attention_mask": all_attention_mask, 
                           "all_labels": all_labels, "all_valid_pos": all_valid_pos, "raw_labels": raw_labels}
            print(f"Saving data to {data_file}")
            torch.save(tensor_data, data_file)
        return self.drop_o(tensor_data, drop_o_ratio)

    def drop_o(self, tensor_data, drop_o_ratio=0):
        if drop_o_ratio == 0:
            return tensor_data
        labels = tensor_data["all_labels"]
        rand_num = torch.rand(labels.size())
        drop_pos = (labels == 0) & (rand_num < drop_o_ratio)
        labels[drop_pos] = -100
        tensor_data["all_labels"] = labels
        return tensor_data
