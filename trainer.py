
import json
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from transformers import (AdamW, RobertaTokenizer, get_linear_schedule_with_warmup)
from tqdm import tqdm
from seqeval.metrics import classification_report
from utils import RoSTERUtils
from model import RoSTERModel
from loss import GCELoss


class RoSTERTrainer(object):

    def __init__(self, args):
        self.args = args
        self.world_size = args.gpus
        self.seed = args.seed
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        self.output_dir = args.output_dir
        self.data_dir = args.data_dir
        self.temp_dir = args.temp_dir
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

        if args.gradient_accumulation_steps < 1:
            raise ValueError(f"Invalid gradient_accumulation_steps parameter: {args.gradient_accumulation_steps}, must be >= 1")

        if args.train_batch_size != 32:
            print(f"Batch size for training is {args.train_batch_size}; 32 is recommended!")
            exit(-1)
        self.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
        self.eval_batch_size = args.eval_batch_size
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.max_seq_length = args.max_seq_length
        
        self.self_train_update_interval = args.self_train_update_interval * args.gradient_accumulation_steps
        self.noise_train_update_interval = args.noise_train_update_interval * args.gradient_accumulation_steps
        self.noise_train_epochs = args.noise_train_epochs
        self.ensemble_train_epochs = args.ensemble_train_epochs
        self.self_train_epochs = args.self_train_epochs

        self.warmup_proportion = args.warmup_proportion
        self.weight_decay = args.weight_decay
        self.q = args.q
        self.tau = args.tau
        
        self.noise_train_lr = args.noise_train_lr
        self.ensemble_train_lr = args.ensemble_train_lr
        self.self_train_lr = args.self_train_lr

        self.tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_model, do_lower_case=False)
        self.processor = RoSTERUtils(self.data_dir, self.tokenizer)
        self.label_map, self.inv_label_map = self.processor.get_label_map(args.tag_scheme)
        self.num_labels = len(self.inv_label_map) - 1  # exclude UNK type
        self.vocab = self.tokenizer.get_vocab()
        self.inv_vocab = {k: v for v, k in self.vocab.items()}
        self.mask_id = self.tokenizer.mask_token_id

        # Prepare model
        self.model = RoSTERModel.from_pretrained(args.pretrained_model, num_labels=self.num_labels-1,
                                                 hidden_dropout_prob=args.dropout, attention_probs_dropout_prob=args.dropout)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if args.do_train:
            tensor_data = self.processor.get_tensor(dataset_name="train", max_seq_length=self.max_seq_length, supervision='dist', drop_o_ratio=0.5)
            
            all_idx = tensor_data["all_idx"]
            all_input_ids = tensor_data["all_input_ids"]
            all_attention_mask = tensor_data["all_attention_mask"]
            all_labels = tensor_data["all_labels"]
            all_valid_pos = tensor_data["all_valid_pos"]
            self.tensor_data = tensor_data
            self.gce_bin_weight = torch.ones_like(all_input_ids).float()
            self.gce_type_weight = torch.ones_like(all_input_ids).float()
            
            self.train_data = TensorDataset(all_idx, all_input_ids, all_attention_mask, all_valid_pos, all_labels)

            print("***** Training stats *****")
            print(f"Num data = {all_input_ids.size(0)}")
            print(f"Batch size = {args.train_batch_size}")

        if args.do_eval:
            tensor_data = self.processor.get_tensor(dataset_name=args.eval_on, max_seq_length=self.max_seq_length, supervision='true')

            all_idx = tensor_data["all_idx"]
            all_input_ids = tensor_data["all_input_ids"]
            all_attention_mask = tensor_data["all_attention_mask"]
            all_labels = tensor_data["all_labels"]
            all_valid_pos = tensor_data["all_valid_pos"]
            self.y_true = tensor_data["raw_labels"]

            eval_data = TensorDataset(all_idx, all_input_ids, all_attention_mask, all_valid_pos, all_labels)
            eval_sampler = SequentialSampler(eval_data)
            self.eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
            
            print("***** Evaluation stats *****")
            print(f"Num data = {all_input_ids.size(0)}")
            print(f"Batch size = {args.eval_batch_size}")

    # prepare model, optimizer and scheduler for training
    def prepare_train(self, lr, epochs):
        model = self.model.to(self.device)
        num_train_steps = int(len(self.train_data)/self.train_batch_size/self.gradient_accumulation_steps) * epochs
        num_train_steps = num_train_steps // self.world_size
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        warmup_steps = int(self.warmup_proportion*num_train_steps)
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)
        model.train()
        return model, optimizer, scheduler

    # training model on distantly-labeled data with noise-robust learning
    def noise_robust_train(self, model_idx=0):
        if os.path.exists(os.path.join(self.temp_dir, f"y_pred_{model_idx}.pt")):
            print(f"\n\n******* Model {model_idx} predictions found; skip training *******\n\n")
            return
        else:
            print(f"\n\n******* Training model {model_idx} *******\n\n")
        model, optimizer, scheduler = self.prepare_train(lr=self.noise_train_lr, epochs=self.noise_train_epochs)
        train_sampler = RandomSampler(self.train_data)
        train_dataloader = DataLoader(self.train_data, sampler=train_sampler, batch_size=self.train_batch_size)
        loss_fct = GCELoss(q=self.q)
        
        i = 0
        for epoch in range(self.noise_train_epochs):
            bin_loss_sum = 0
            type_loss_sum = 0
            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
                if (i+1) % self.noise_train_update_interval == 0:
                    self.update_weights(model)
                    model.train()
                    print(f"bin_loss: {round(bin_loss_sum/self.noise_train_update_interval,5)}; type_loss: {round(type_loss_sum/self.noise_train_update_interval,5)}")
                    bin_loss_sum = 0
                    type_loss_sum = 0
                idx, input_ids, attention_mask, valid_pos, labels = tuple(t.to(self.device) for t in batch)
                bin_weights = self.gce_bin_weight[idx].to(self.device)
                type_weights = self.gce_type_weight[idx].to(self.device)

                max_len = attention_mask.sum(-1).max().item()
                input_ids, attention_mask, valid_pos, labels, bin_weights, type_weights = tuple(t[:, :max_len] for t in \
                        (input_ids, attention_mask, valid_pos, labels, bin_weights, type_weights))
                
                type_logits, bin_logits = model(input_ids, attention_mask, valid_pos)
                
                labels = labels[valid_pos > 0]
                bin_weights = bin_weights[valid_pos > 0]
                type_weights = type_weights[valid_pos > 0]

                bin_labels = labels.clone()
                bin_labels[labels > 0] = 1
                type_labels = labels - 1
                type_labels[type_labels < 0] = -100

                type_loss = loss_fct(type_logits.view(-1, self.num_labels-1), type_labels.view(-1), type_weights)
                type_loss_sum += type_loss.item()

                bin_loss = loss_fct(bin_logits.view(-1, 1), bin_labels.view(-1), bin_weights)
                bin_loss_sum += bin_loss.item()
                
                loss = type_loss + bin_loss
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps
                
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                if (step+1) % self.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                i += 1
            
            if self.args.do_eval:
                y_pred, _ = self.eval(model, self.eval_dataloader)
                print(f"\n****** Evaluating on {self.args.eval_on} set: ******\n")
                self.performance_report(self.y_true, y_pred)
                
        eval_sampler = SequentialSampler(self.train_data)
        eval_dataloader = DataLoader(self.train_data, sampler=eval_sampler, batch_size=self.eval_batch_size)
        y_pred, pred_probs = self.eval(model, eval_dataloader)
        torch.save({"pred_probs": pred_probs}, os.path.join(self.temp_dir, f"y_pred_{model_idx}.pt"))

    # assign 0/1 weights to each training token based on whether the model prediction agrees with the distant label (noisy label removal)
    def update_weights(self, model):
        train_sampler = RandomSampler(self.train_data)
        train_dataloader = DataLoader(self.train_data, sampler=train_sampler, batch_size=self.train_batch_size)
        model.eval()
        for batch in tqdm(train_dataloader, desc="Updating training data weights"):
            idx, input_ids, attention_mask, valid_pos, labels = tuple(t.to(self.device) for t in batch)
            
            type_weight = torch.ones_like(input_ids).float()
            bin_weight = torch.ones_like(input_ids).float()

            max_len = attention_mask.sum(-1).max().item()
            input_ids, attention_mask, valid_pos, labels = tuple(t[:, :max_len] for t in \
                        (input_ids, attention_mask, valid_pos, labels))
            with torch.no_grad():
                type_logits, bin_logits = model(input_ids, attention_mask, valid_pos)
                valid_idx = labels != -100
                type_pos = labels > 0
                labels = labels[valid_pos > 0]
                type_logits = type_logits[labels > 0]
                bin_logits = bin_logits[labels != -100]
                bin_labels = labels.clone()
                bin_labels[labels > 0] = 1
                bin_labels = bin_labels[labels != -100]
                labels = labels[labels > 0]
                type_labels = labels - 1

                type_pred = F.softmax(type_logits, dim=-1)
                type_pred_label = torch.gather(type_pred, dim=-1, index=type_labels.unsqueeze(-1)).squeeze(-1)
                entity_prob = torch.sigmoid(bin_logits)
                bin_pred = torch.cat((1-entity_prob, entity_prob), dim=-1)
                bin_pred_label = torch.gather(bin_pred, dim=-1, index=bin_labels.unsqueeze(-1)).squeeze(-1)
                
                condition = type_pred_label > self.tau
                type_weight[:, :max_len][(valid_pos > 0) & type_pos] = condition.to(type_weight)
                condition = (bin_pred_label > self.tau) | (bin_labels == 1)
                bin_weight[:, :max_len][(valid_pos > 0) & valid_idx] = condition.to(bin_weight)

            self.gce_bin_weight[idx] = bin_weight.cpu()
            self.gce_type_weight[idx] = type_weight.cpu()
        
        # check if there are too few training tokens for any entity type classes
        remove_label_pos = self.gce_type_weight == 0
        for i in range(1, self.num_labels):
            type_label_pos = self.tensor_data["all_labels"] == i
            remove_type_num = (type_label_pos & remove_label_pos.cpu()).sum().item()
            remove_type_frac = remove_type_num / type_label_pos.sum().item()
            if remove_type_frac > 0.9:
                self.gce_type_weight[type_label_pos] = 1

    # compute ensembled predictions
    def ensemble_pred(self, fild_dir):
        pred_prob_list = []
        for f in os.listdir(fild_dir):
            if f.startswith('y_pred'):
                pred = torch.load(os.path.join(fild_dir, f))
                pred_prob_list.append(pred["pred_probs"])
        ensemble_probs = []
        for i in range(len(pred_prob_list[0])):
            ensemble_prob_sent = []
            for j in range(len(pred_prob_list[0][i])):
                all_pred_probs = torch.cat([pred_prob_list[k][i][j].unsqueeze(0) for k in range(len(pred_prob_list))], dim=0)
                ensemble_prob_sent.append(torch.mean(all_pred_probs, dim=0, keepdim=True))
            ensemble_probs.append(torch.cat(ensemble_prob_sent, dim=0))
        ensemble_preds = []
        for pred_prob in ensemble_probs:
            preds = pred_prob.argmax(dim=-1)
            ensemble_preds.append([self.inv_label_map[pred.item()] for pred in preds])
        all_valid_pos = self.tensor_data["all_valid_pos"]
        ensemble_label = -100 * torch.ones(all_valid_pos.size(0), all_valid_pos.size(1), self.num_labels)
        ensemble_label[all_valid_pos > 0] = torch.cat(ensemble_probs, dim=0)
        self.ensemble_label = ensemble_label

    # train an ensembled model
    def ensemble_train(self):
        if os.path.exists(os.path.join(self.temp_dir, "ensemble_model.pt")):
            print(f"\n\n******* Ensemble model found; skip training *******\n\n")
            return
        else:
            print("\n\n******* Training ensembled model *******\n\n")
        model, optimizer, scheduler = self.prepare_train(lr=self.ensemble_train_lr, epochs=self.ensemble_train_epochs)

        all_input_ids = self.tensor_data["all_input_ids"]
        all_attention_mask = self.tensor_data["all_attention_mask"]
        all_valid_pos = self.tensor_data["all_valid_pos"]
        all_soft_labels = self.ensemble_label
        ensemble_train_data = TensorDataset(all_input_ids, all_attention_mask, all_valid_pos, all_soft_labels)
        train_sampler = RandomSampler(ensemble_train_data)
        train_dataloader = DataLoader(ensemble_train_data, sampler=train_sampler, batch_size=self.train_batch_size)
        
        for epoch in range(self.ensemble_train_epochs):
            type_loss_sum = 0
            bin_loss_sum = 0
            model.train()
            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
                input_ids, attention_mask, valid_pos, labels = tuple(t.to(self.device) for t in batch)

                max_len = attention_mask.sum(-1).max().item()
                input_ids, attention_mask, valid_pos, labels = tuple(t[:, :max_len] for t in \
                        (input_ids, attention_mask, valid_pos, labels))
                
                labels = labels[valid_pos > 0]
                target_type = labels
                target_type = target_type[:,1:]
                target_type = target_type / target_type.sum(dim=-1, keepdim=True)

                type_logits, bin_logits = model(input_ids, attention_mask, valid_pos)
                
                loss_fct = nn.KLDivLoss(reduction='sum')
                preds = F.log_softmax(type_logits, dim=-1)
                type_loss = loss_fct(preds, target_type)
                if type_logits.size(0) > 0:
                    type_loss = type_loss / type_logits.size(0)
                    type_loss_sum += type_loss.item()
                
                bin_labels = 1 - labels[:,0]
                loss_fct = nn.BCEWithLogitsLoss()
                bin_loss = loss_fct(bin_logits.view(-1), bin_labels.view(-1))
                bin_loss_sum += bin_loss.item()

                loss = type_loss + bin_loss
                
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                if (step+1) % self.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
            
            print(f"bin_loss: {round(bin_loss_sum/step,5)}; type_loss: {round(type_loss_sum/step,5)}")
            
            if self.args.do_eval:
                y_pred, _ = self.eval(model, self.eval_dataloader)
                print(f"\n****** Evaluating on {self.args.eval_on} set: ******\n")
                self.performance_report(self.y_true, y_pred)
        
        self.save_model(model, "ensemble_model.pt", self.temp_dir)

    # use pre-trained RoBERTa to create contextualized augmentations given original sequences
    def aug(self, mask_prob=0.15, save_name="aug.pt"):
        model = self.model.to(self.device)
        model.eval()
        train_sampler = RandomSampler(self.train_data)
        train_dataloader = DataLoader(self.train_data, sampler=train_sampler, batch_size=self.eval_batch_size)
        all_aug_input_ids = []
        all_idx = []
        for batch in tqdm(train_dataloader, desc="Creating augmentations"):
            idx, input_ids, attention_mask, valid_pos, _ = tuple(t.to(self.device) for t in batch)
            aug_input_ids = input_ids.clone()
            
            mask_pos = torch.rand(input_ids.size(), device=self.device) < mask_prob
            orig_ids = input_ids[valid_pos > 0]
            input_ids[mask_pos] = self.mask_id
            with torch.no_grad():
                mlm_logits = model.mlm_pred(input_ids, attention_mask, valid_pos)
                top_logits, top_idx = mlm_logits.topk(k=5, dim=-1)
                sample_probs = F.softmax(top_logits, dim=-1)
                sampled_token_idx = torch.multinomial(sample_probs, 1).view(-1)
                sampled_ids = top_idx[torch.arange(top_idx.size(0)), sampled_token_idx]
            for i in range(len(sampled_ids)):
                sampled_token = self.inv_vocab[sampled_ids[i].item()]
                orig_token = self.inv_vocab[orig_ids[i].item()]
                if (sampled_token.startswith('Ġ') ^ orig_token.startswith('Ġ')) or sampled_token == 'Ġ' or orig_token == 'Ġ' \
                    or (sampled_token.split('Ġ')[-1][0].isupper() ^ orig_token.split('Ġ')[-1][0].isupper()):
                    sampled_ids[i] = orig_ids[i]
            
            aug_input_ids[valid_pos > 0] = sampled_ids
            all_aug_input_ids.append(aug_input_ids)
            all_idx.append(idx)
        all_aug_input_ids = torch.cat(all_aug_input_ids)
        all_idx = torch.cat(all_idx)

        all_aug_res = {}
        for data_idx, aug_input_ids in zip(all_idx, all_aug_input_ids):
            all_aug_res[data_idx.item()] = aug_input_ids
        aug_input_ids = []
        for i in range(len(all_aug_res)):
            aug_input_ids.append(all_aug_res[i].unsqueeze(0))
        aug_input_ids = torch.cat(aug_input_ids, dim=0)
        torch.save(aug_input_ids, os.path.join(self.temp_dir, save_name))

    # compute soft labels for self-training on entity type classes
    def soft_labels(self, model, entity_threshold=0.8):
        train_sampler = RandomSampler(self.train_data)
        train_dataloader = DataLoader(self.train_data, sampler=train_sampler, batch_size=self.eval_batch_size)
        model.eval()

        type_preds = []
        indices = []
        for batch in tqdm(train_dataloader, desc="Computing soft labels"):
            idx, input_ids, attention_mask, valid_pos, labels = tuple(t.to(self.device) for t in batch)
            type_distrib = torch.zeros(input_ids.size(0), self.max_seq_length, self.num_labels-1).to(self.device)

            max_len = attention_mask.sum(-1).max().item()
            input_ids, attention_mask, valid_pos, labels = tuple(t[:, :max_len] for t in \
                        (input_ids, attention_mask, valid_pos, labels))
            with torch.no_grad():
                type_logits, bin_logits = model(input_ids, attention_mask, valid_pos)
                type_pred = F.softmax(type_logits, dim=-1)
                entity_prob = torch.sigmoid(bin_logits)
                type_pred[entity_prob.squeeze() < entity_threshold] = 0
                type_distrib[:, :max_len][valid_pos > 0] = type_pred
                type_preds.append(type_distrib)

            indices.append(idx)
        
        type_preds = torch.cat(type_preds, dim=0)
        all_idx = torch.cat(indices)

        type_distribution = torch.zeros(len(self.train_data), self.max_seq_length, self.num_labels-1)
        for idx, type_pred in zip(all_idx, type_preds):
            type_distribution[idx] = type_pred

        type_distribution = type_distribution.view(-1, type_distribution.size(-1))
        valid_rows = type_distribution.sum(dim=-1) > 0
        weight = type_distribution[valid_rows]**2 / torch.sum(type_distribution[valid_rows], dim=0)
        target_distribution = (weight.t() / torch.sum(weight, dim=-1)).t()
        type_distribution[valid_rows] = target_distribution
        type_distribution = type_distribution.view(len(self.train_data), self.max_seq_length, self.num_labels-1)
        
        return type_distribution

    # self-training with augmentation
    def self_train(self):
        if os.path.exists(os.path.join(self.output_dir, "final_model.pt")):
            print(f"\n\n******* Final model found; skip training *******\n\n")
            return
        else:
            print("\n\n******* Self-training *******\n\n")
        self.load_model("ensemble_model.pt", self.temp_dir)
        model, optimizer, scheduler = self.prepare_train(lr=self.self_train_lr, epochs=self.self_train_epochs)

        all_idx = self.tensor_data["all_idx"]
        all_input_ids = self.tensor_data["all_input_ids"]
        all_attention_mask = self.tensor_data["all_attention_mask"]
        all_valid_pos = self.tensor_data["all_valid_pos"]
        all_soft_labels = self.ensemble_label

        i = 0
        for epoch in range(self.self_train_epochs):
            type_loss_sum = 0
            bin_loss_sum = 0
            aug_loss_sum = 0

            self.aug(mask_prob=0.15)
            data_file = os.path.join(self.temp_dir, "aug.pt")
            all_aug_input_ids = torch.load(data_file)
            aug_train_data = TensorDataset(all_idx, all_input_ids, all_aug_input_ids, all_attention_mask, all_valid_pos, all_soft_labels)
            train_sampler = RandomSampler(aug_train_data)
            train_dataloader = DataLoader(aug_train_data, sampler=train_sampler, batch_size=self.train_batch_size)

            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
                if i % self.self_train_update_interval == 0:
                    type_distribution = self.soft_labels(model)
                    model.train()
                    if i != 0:
                        print(f"bin_loss: {round(bin_loss_sum/self.self_train_update_interval,5)}; type_loss: {round(type_loss_sum/self.self_train_update_interval,5)}; aug_loss: {round(aug_loss_sum/self.self_train_update_interval,5)}")
                    type_loss_sum = 0
                    bin_loss_sum = 0
                    aug_loss_sum = 0
                
                idx, input_ids, aug_input_ids, attention_mask, valid_pos, labels = tuple(t.to(self.device) for t in batch)
                target_type = type_distribution[idx].to(self.device)

                max_len = attention_mask.sum(-1).max().item()
                input_ids, aug_input_ids, attention_mask, valid_pos, labels, target_type = tuple(t[:, :max_len] for t in \
                        (input_ids, aug_input_ids, attention_mask, valid_pos, labels, target_type))

                type_logits, bin_logits = model(input_ids, attention_mask, valid_pos)
                
                valid_type = target_type[valid_pos > 0].sum(dim=-1) > 0
                type_logits = type_logits[valid_type]
                target_type = target_type[valid_pos > 0][valid_type]
                loss_fct = nn.KLDivLoss(reduction='sum')
                preds = F.log_softmax(type_logits, dim=-1)
                orig_pred_type = preds.argmax(-1)
                type_loss = loss_fct(preds, target_type)
                if type_logits.size(0) > 0:
                    type_loss = type_loss / type_logits.size(0)
                    type_loss_sum += type_loss.item()

                bin_loss_fct = nn.BCEWithLogitsLoss()
                labels = labels[valid_pos > 0]
                bin_labels = 1 - labels[:,0]
                bin_loss = bin_loss_fct(bin_logits.view(-1), bin_labels.float())
                bin_loss_sum += bin_loss.item()
                
                aug_logits, _, = model(aug_input_ids, attention_mask, valid_pos)
                aug_logits = aug_logits[valid_type]

                preds = F.log_softmax(aug_logits, dim=-1)
                aug_pred_type = preds.argmax(-1)
                agree_pos = aug_pred_type == orig_pred_type
                preds = preds[agree_pos]
                target_type = target_type[agree_pos]
                aug_loss = loss_fct(preds, target_type)
                if preds.size(0) > 0:
                    aug_loss = aug_loss / preds.size(0)
                    aug_loss_sum += aug_loss.item()

                loss = type_loss + bin_loss + aug_loss
                
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                if (step+1) % self.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                i += 1
            
            if self.args.do_eval:
                y_pred, _ = self.eval(model, self.eval_dataloader)
                print(f"\n****** Evaluating on {self.args.eval_on} set: ******\n")
                self.performance_report(self.y_true, y_pred)
        
        self.save_model(model, "final_model.pt", self.output_dir)

    # obtain model predictions on a given dataset
    def eval(self, model, eval_dataloader):
        model = model.to(self.device)
        model.eval()
        y_pred = []
        pred_probs = []
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            _, input_ids, attention_mask, valid_pos, _ = tuple(t.to(self.device) for t in batch)

            max_len = attention_mask.sum(-1).max().item()
            input_ids, attention_mask, valid_pos = tuple(t[:, :max_len] for t in \
                        (input_ids, attention_mask, valid_pos))

            with torch.no_grad():
                logits, bin_logits = model(input_ids, attention_mask, valid_pos)
                entity_prob = torch.sigmoid(bin_logits)
                type_prob = F.softmax(logits, dim=-1) * entity_prob
                non_type_prob = 1 - entity_prob
                type_prob = torch.cat([non_type_prob, type_prob], dim=-1)
                
                preds = torch.argmax(type_prob, dim=-1)
                preds = preds.cpu().numpy()
                pred_prob = type_prob.cpu()

            num_valid_tokens = valid_pos.sum(dim=-1)
            i = 0
            for j in range(len(num_valid_tokens)):
                pred_probs.append(pred_prob[i:i+num_valid_tokens[j]])
                y_pred.append([self.inv_label_map[pred] for pred in preds[i:i+num_valid_tokens[j]]])
                i += num_valid_tokens[j]

        return y_pred, pred_probs

    # print out ner performance given ground truth and model prediction
    def performance_report(self, y_true, y_pred):
        for i in range(len(y_true)):
            if len(y_true[i]) > len(y_pred[i]):
                print(f"Warning: Sequence {i} is truncated for eval! ({len(y_pred[i])}/{len(y_true[i])})")
                y_pred[i] = y_pred[i] + ['O'] * (len(y_true[i])-len(y_pred[i]))
        report = classification_report(y_true, y_pred, digits=3)
        print(report)

    # save model, tokenizer, and configs to directory
    def save_model(self, model, model_name, save_dir):
        print(f"Saving {model_name} ...")
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), os.path.join(save_dir, model_name))
        self.tokenizer.save_pretrained(save_dir)
        model_config = {"max_seq_length": self.max_seq_length, 
                        "num_labels": self.num_labels, 
                        "label_map": self.label_map}
        json.dump(model_config, open(os.path.join(save_dir, "model_config.json"), "w"))

    # load model from directory
    def load_model(self, model_name, load_dir):
        self.model.load_state_dict(torch.load(os.path.join(load_dir, model_name)))
