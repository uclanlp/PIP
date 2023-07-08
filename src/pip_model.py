import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BartForConditionalGeneration, BartConfig
from transformers import BartForConditionalGeneration, BartModel, BertModel
from prefix2 import PrefixGenBartForPrefixControledGeneration, PrefixGenBartForConditionalGeneration, PrefixGenBartEncoder
from transformers.adapters import PrefixTuningConfig
from transformers.models.bart.modeling_bart import BartEncoder, BaseModelOutput, _expand_mask
import ipdb
import math
from nltk.tree import Tree, ParentedTree
from calc_prefix_vocab import find_vocab_size
import pickle

class ParaphraseModel(nn.Module):
    def __init__(self, config, tokenizer, device, debug=True):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.parse_vocab_size = 83
        self.input_size = 768 # Bert Hidden size

        self.bart_config = BartConfig.from_pretrained('facebook/bart-base')
        
        self.model = PrefixGenBartForConditionalGeneration(self.bart_config).from_pretrained("facebook/bart-base")
        self.model.resize_token_embeddings(len(self.tokenizer))

        assert self.config.prefix_model_type == "self"
        self.prefix_tokenizer = tokenizer
        # cos sim prefix loss
        self.prefix_criterion = nn.CosineSimilarity(dim=1)
        self.prefix_config = PrefixTuningConfig

        self.n_layers = 6
        self.n_heads = 12
        self.n_embd_per_head = self.input_size // self.n_heads
        self.prefix_length = self.config.prefix_length

        if self.config.prefix_type in ["attention0", "ptuning"]:
            # # self.attention = nn.MultiheadAttention(embed_dim = 1, num_heads = 1)
            # self.attention = nn.MultiheadAttention(embed_dim = self.input_size, num_heads = self.n_heads)
            self.register_buffer("prefix_ids", torch.arange(self.config.prefix_length).expand((1, -1))) # (1, prefix_len)
            self.wte_1 = nn.Embedding(self.config.prefix_length, self.input_size)
            self.wte_2 = nn.Embedding(self.config.prefix_length, self.input_size)
            self.wte_3 = nn.Embedding(self.config.prefix_length, self.input_size)
            # self.wte_1 = nn.Embedding(self.parse_vocab_size, self.input_size)
            # self.wte_2 = nn.Embedding(self.parse_vocab_size, self.input_size)
            # self.wte_3 = nn.Embedding(self.parse_vocab_size, self.input_size)
    
        if self.config.prefix_type == "attention0":
            # # self.linear = nn.Linear(self.prefix_length, self.prefix_length)
            # # self.linear_1 = nn.Linear(self.input_size, self.input_size)
            # # self.linear_2 = nn.Linear(self.input_size, self.input_size)
            # self.linear = nn.Linear(self.input_size, self.input_size)
            # # self.mu = nn.Parameter(torch.Tensor(1),requires_grad=True)
            # self.mu = 1
            
            self.control_trans_1 = nn.Sequential(
                nn.Linear(self.input_size, self.prefix_config.bottleneck_size),
                nn.Tanh(),
                # nn.Linear(self.prefix_config.bottleneck_size, self.n_layers * 2 * self.input_size), # original
                nn.Linear(self.prefix_config.bottleneck_size, self.n_layers * 2 * self.input_size), # enc+dec
            )
            self.control_trans_2 = nn.Sequential(
                nn.Linear(self.input_size, self.prefix_config.bottleneck_size),
                nn.Tanh(),
                # nn.Linear(self.prefix_config.bottleneck_size, self.n_layers * 2 * self.input_size), # original
                nn.Linear(self.prefix_config.bottleneck_size, self.n_layers * 2 * self.input_size), # enc+dec
            )
            self.control_trans_3 = nn.Sequential(
                nn.Linear(self.input_size, self.prefix_config.bottleneck_size),
                nn.Tanh(),
                # nn.Linear(self.prefix_config.bottleneck_size, self.n_layers * 2 * self.input_size), # original
                nn.Linear(self.prefix_config.bottleneck_size, self.n_layers * 2 * self.input_size), # enc+dec
            )
        elif self.config.prefix_type == "ptuning":
            self.control_trans_1 = nn.Sequential(
                nn.Linear(self.input_size, self.prefix_config.bottleneck_size),
                nn.Tanh(),
                nn.Linear(self.prefix_config.bottleneck_size, 2 * self.prefix_config.bottleneck_size),
                # nn.Linear(self.prefix_config.bottleneck_size, self.n_layers * 2 * self.input_size), # original
                nn.Linear(2 * self.prefix_config.bottleneck_size, self.n_layers * 2 * self.input_size), # enc+dec
            )
            self.control_trans_2 = nn.Sequential(
                nn.Linear(self.input_size, self.prefix_config.bottleneck_size),
                nn.Tanh(),
                nn.Linear(self.prefix_config.bottleneck_size, 2 * self.prefix_config.bottleneck_size),
                # nn.Linear(self.prefix_config.bottleneck_size, self.n_layers * 2 * self.input_size), # original
                nn.Linear(2 * self.prefix_config.bottleneck_size, self.n_layers * 2 * self.input_size), # enc+dec
            )
            self.control_trans_3 = nn.Sequential(
                nn.Linear(self.input_size, self.prefix_config.bottleneck_size),
                nn.Tanh(),
                nn.Linear(self.prefix_config.bottleneck_size, 2 * self.prefix_config.bottleneck_size),
                # nn.Linear(self.prefix_config.bottleneck_size, self.n_layers * 2 * self.input_size), # original
                nn.Linear(2 * self.prefix_config.bottleneck_size, self.n_layers * 2 * self.input_size), # enc+dec
            )
    
        self.dropout = nn.Dropout(self.prefix_config.dropout)

        self.debug = debug
        self.device = device
        if self.debug:
            self.show_demo_examples = True

    def resolve_attn_mask(self, old_attn_mask):
        new_attn_mask = F.pad(old_attn_mask, (30, 0, 0, 0), "constant", 1)
        return new_attn_mask
    
    def resolve_len(self, input, tgt_len):
        size = input.size()[1]
        new_input = torch.empty(input.size()[0], tgt_len, input.size()[2])

        if size < tgt_len:
            input = F.pad(input, (0, tgt_len - size), "constant", 0)
        else:
            for i in range(input.size()[0]):
                l = [j for j in range(size)]
                keep = random.sample(l, tgt_len)
                for k in range(tgt_len):
                    new_input[i][k][:] = input[i][keep[k]][:]
        return new_input.to(self.device)
    
    def resolve_synt_tok(self, synt):
        synt1 = synt.split()
        new_synt = []
        for s in synt1:
            i, j = 0, len(s) - 1
            while s[i] == "(":
                new_synt.append(s[i])
                i += 1
            right_ct = 0
            while s[j] == ")":
                right_ct += 1
                j -= 1
            new_synt.append(s[i:j + 1])
            new_synt += [")" for i in range(right_ct)]
        return ' '.join(new_synt)

    def get_synt_tok(self, prefix_inputs, prefix_tok_word2idx):
        prefix_inputs = [p.split() for p in prefix_inputs] # bsz, seq_len
        # max_len = max(len(p) for p in prefix_inputs)
        # prefix_tok = torch.ones(len(prefix_inputs), self.prefix_length)
        prefix_tok = torch.zeros(len(prefix_inputs), self.prefix_length)
        for i in range(len(prefix_inputs)):
            synt = prefix_inputs[i]
            for j in range(len(synt)):
                prefix_tok[i][j] = prefix_tok_word2idx[synt[j]]
        
        return prefix_tok.to(self.device) #.to(torch.long)

    def process_data(self, src_sents, src_synts, tgt_synts, tgt_sents=None):
        # encoder inputs
        if self.config.use_enc_src_parse:
            input_texts = [f"{src_sent} {self.config.sep_token} {src_synt} {self.config.sep_token} {tgt_synt}" for src_sent, src_synt, tgt_synt in zip(src_sents, src_synts, tgt_synts)]
        else:
            input_texts = [f"{src_sent} {self.config.sep_token} {tgt_synt}" for src_sent, tgt_synt in zip(src_sents, tgt_synts)]
        inputs = self.tokenizer(input_texts, return_tensors='pt', padding=True)

        inputs = inputs.to(self.device)

        enc_idxs = inputs['input_ids']
        enc_attn = inputs['attention_mask']
        
        enc_idxs = enc_idxs.cuda()
        enc_attn = enc_attn.cuda()
        
        if tgt_sents is None:
            return enc_idxs, enc_attn, None, None, None
        
        # decoder inputs
        if self.config.use_dec_tgt_parse:
            output_texts = [f"{tgt_synt} {self.config.sep_token} {tgt_sent}" for tgt_synt, tgt_sent in zip(tgt_synts, tgt_sents)]
        else:
            output_texts = tgt_sents
        outputs = self.tokenizer(output_texts, return_tensors='pt', padding=True)

        outputs = outputs.to(self.device)
        
        # dec_idxs = outputs['input_ids']
        # batch_size = dec_idxs.size(0)
        # dec_idxs[:, 0] = self.tokenizer.eos_token_id
        # dec_attn = outputs['attention_mask']
        
        batch_size = enc_idxs.size(0)
        
        padding = torch.ones((batch_size, 1), dtype=torch.long, device = self.device)
        padding[:] = self.tokenizer.eos_token_id
        dec_idxs = torch.cat((padding, outputs['input_ids']), dim=1)
        dec_attn = torch.cat((torch.ones((batch_size, 1), dtype=torch.long, device = self.device), outputs['attention_mask']), dim=1)
            
        # labels
        padding = torch.ones((batch_size, 1), dtype=torch.long, device = self.device)
        padding[:] = self.tokenizer.pad_token_id
        raw_lbl_idxs = torch.cat((dec_idxs[:, 1:], padding), dim=1)
        lbl_attn = torch.cat((dec_attn[:, 1:], torch.zeros((batch_size, 1), dtype=torch.long, device = self.device)), dim=1)
        lbl_idxs = raw_lbl_idxs.masked_fill(lbl_attn==0, -100) # ignore padding
        
        dec_idxs = dec_idxs.cuda()
        dec_attn = dec_attn.cuda()
        lbl_idxs = lbl_idxs.cuda()
        
        if self.show_demo_examples:
            print()
            for i in range(3):
                print(f"IN:\n {input_texts[i]}")
                print(f"OUT:\n {output_texts[i]}")
            self.show_demo_examples = False
        
        return enc_idxs, enc_attn, dec_idxs, dec_attn, lbl_idxs

    def process_pip_data(self, src_sents, src_synts, tgt_synts, tgt_sents=None):
        if self.config.model_type == "prefix_reg": # regular prefix
            prefix = [[i for i in range(self.config.prefix_length)]for j in range(len(tgt_synts))]
            prefix = torch.FloatTensor(prefix).to(self.device)
        else:
            # prefix
            # prefix_inputs = [f"{tgt_synt}" for tgt_synt in tgt_synts]
            prefix_inputs = [f"{self.resolve_synt_tok(tgt_synt)}" for tgt_synt in tgt_synts]

            # prefix_inputs = self.prefix_tokenizer(prefix_inputs, return_tensors='pt', padding=True) # (bsz, seq_len) # orginal
            if self.config.prefix_type == "attention0":
                # for ptuning-based 
                # prefix_inputs = self.get_synt_tok(prefix_inputs, self.prefix_tok_word2idx) # (bsz, prefix_len)
                # prefix_inputs = self.prefix_tokenizer(prefix_inputs, return_tensors='pt', padding=True) # (bsz, seq_len) # orginal

                # for cossim-based p tuning
                prefix_inputs = self.prefix_tokenizer(prefix_inputs, return_tensors='pt', max_length = self.config.prefix_length, padding='max_length')
                prefix_inputs = prefix_inputs['input_ids'].to(self.device)

                # self.enc_outputs_1 = enc_outputs.hidden_states[0].to(self.device)
                # self.enc_outputs_2 = enc_outputs.last_hidden_state.to(self.device)

                # for encoder input cos sim
                # enc_outputs = self.model.model.encoder(prefix_inputs,output_hidden_states=True)
                # self.enc_inputs = enc_outputs.hidden_states[0].to(self.device)
                # for encoder output cos sim
                enc_outputs = self.model.model.encoder(prefix_inputs,output_hidden_states=True)
                self.enc_outputs = enc_outputs.last_hidden_state.to(self.device)
            
            if self.config.prefix_type not in ["attention0", "ptuning"]:
                prefix_inputs = self.model.model.encoder(prefix_inputs)["last_hidden_state"]

            if self.config.prefix_type == "attention0":
                # original
                # prefix_inputs = torch.cat([self.prefix_ids.expand(prefix_inputs.size()[0], -1), prefix_inputs], dim=1) # (batch_size, prefix_length + seq_length, input_size)
                # prefix_inputs = self.attention(prefix_inputs.unsqueeze(-1).permute(1,0,2), prefix_inputs.unsqueeze(-1).permute(1,0,2), prefix_inputs.unsqueeze(-1).permute(1,0,2))[0].permute(1,0,2)[:, :self.config.prefix_length, :].squeeze(2)
                # prefix_1 = self.wte_1(self.prefix_ids.expand(prefix_inputs.size()[0], -1)) # (batch_size, prefix_length, input_size)
                # prefix_2 = self.wte_2(self.prefix_ids.expand(prefix_inputs.size()[0], -1)) # (batch_size, prefix_length, input_size)
                # prefix_3 = self.wte_3(self.prefix_ids.expand(prefix_inputs.size()[0], -1)) # (batch_size, prefix_length, input_size)
                
                # 1 for mu
                # prefix_inputs = (1 - self.mu) * self.prefix_ids.expand(prefix_inputs.size()[0], -1) + self.mu * prefix_inputs # original
                # 2
                prefix_inputs = self.prefix_ids.expand(len(prefix_inputs), -1) # for prefix tuning
                # 3 for linear
                # prefix_inputs = self.prefix_ids.expand(prefix_inputs.size()[0], -1) + self.linear(prefix_inputs)
                # prefix_inputs = prefix_inputs - torch.min(prefix_inputs)
                # prefix_inputs = prefix_inputs * (self.prefix_length - 1) // torch.max(prefix_inputs)
                # 4 for cross similarity
                # prefix_inputs = self.attention(prefix_inputs.unsqueeze(-1).permute(1,0,2).to(torch.float32), self.prefix_ids.expand(prefix_inputs.size()[0], -1).unsqueeze(-1).permute(1,0,2).to(torch.float32), self.prefix_ids.expand(prefix_inputs.size()[0], -1).unsqueeze(-1).permute(1,0,2).to(torch.float32))[0].permute(1,0,2).squeeze(2)
                # prefix_inputs = prefix_inputs - torch.min(prefix_inputs)
                # prefix_inputs = prefix_inputs * (self.prefix_length - 1) // torch.max(prefix_inputs)

                prefix_inputs = prefix_inputs.to(torch.long)
                # print('prefix', prefix_inputs[0])
                prefix_1 = self.wte_1(prefix_inputs) # (batch_size, prefix_length, input_size)
                prefix_2 = self.wte_2(prefix_inputs) # (batch_size, prefix_length, input_size)
                prefix_3 = self.wte_3(prefix_inputs) # (batch_size, prefix_length, input_size)

                embds_1 = prefix_1
                key_values_1 = self.control_trans_1(embds_1) #embds)  # batch_size x prefix_length x n_layers*2*input_size
                key_values_1 = key_values_1.view(
                    key_values_1.size()[0], self.prefix_length, self.n_layers * 2, self.n_heads, self.n_embd_per_head # original
                )  # *2 for key and value
                key_values_1 = self.dropout(key_values_1) # n_layers * (2 x batch_size x n_heads x prefix_length x n_embd_per_head)

                ##
                embds_2 = prefix_2
                key_values_2 = self.control_trans_2(embds_2) #embds)  # batch_size x prefix_length x n_layers*2*input_size
                key_values_2 = key_values_2.view(
                    key_values_2.size()[0], self.prefix_length, self.n_layers * 2, self.n_heads, self.n_embd_per_head # original
                )  # *2 for key and value
                key_values_2 = self.dropout(key_values_2) # n_layers * (2 x batch_size x n_heads x prefix_length x n_embd_per_head)

                ### 
                embds_3 = prefix_3
                key_values_3 = self.control_trans_3(embds_3) #embds)  # batch_size x prefix_length x n_layers*2*input_size
                key_values_3 = key_values_3.view(
                    key_values_3.size()[0], self.prefix_length, self.n_layers * 2, self.n_heads, self.n_embd_per_head # original
                )  # *2 for key and value
                key_values_3 = self.dropout(key_values_3) # n_layers * (2 x batch_size x n_heads x prefix_length x n_embd_per_head)

                prefix_dict = {} # need: (bz, n_encoder_layers, prefix_len, head_num, dim)

                # For regular (only enc) prefix
                key_values_1 = key_values_1.permute(0, 2, 1, 3, 4).split(self.n_layers, dim = 1)

                # # for cos sim based p-tuning
                # batch_size = key_values_3.size()[0]
                # # for modified cosine similarity
                # # for enc input value cos sim
                # # prefix_enc_inputs = torch.cat([key_values_1[1][:,0,:,:,:].squeeze(1).reshape(batch_size, self.prefix_length, -1), self.enc_inputs], dim=1) # (batch_size, prefix_length + seq_length, input_size)
                # # self.prefix_enc_inputs = self.attention(prefix_enc_inputs.permute(1,0,2), prefix_enc_inputs.permute(1,0,2), prefix_enc_inputs.permute(1,0,2))[0].permute(1,0,2)[:, :self.config.prefix_length, :].to(self.device) # for padded prefix
                # # self.prefix_enc_inputs = self.linear(self.prefix_enc_inputs)
                # # for enc input key cos sim
                # # prefix_enc_inputs = torch.cat([key_values_1[0][:,0,:,:,:].squeeze(1).reshape(batch_size, self.prefix_length, -1), self.enc_inputs], dim=1) # (batch_size, prefix_length + seq_length, input_size)
                # # self.prefix_enc_inputs = self.attention(prefix_enc_inputs.permute(1,0,2), prefix_enc_inputs.permute(1,0,2), prefix_enc_inputs.permute(1,0,2))[0].permute(1,0,2)[:, :self.config.prefix_length, :].to(self.device) # for padded prefix
                # # self.prefix_enc_inputs = self.linear(self.prefix_enc_inputs)
                # # for enc output cos sim
                # # prefix_enc_outputs = torch.cat([key_values_1[1][:,5,:,:,:].squeeze(1).reshape(batch_size, self.prefix_length, -1), self.enc_outputs], dim=1) # (batch_size, prefix_length + seq_length, input_size)
                # # self.prefix_enc_outputs = self.attention(prefix_enc_outputs.permute(1,0,2), prefix_enc_outputs.permute(1,0,2), prefix_enc_outputs.permute(1,0,2))[0].permute(1,0,2)[:, :self.config.prefix_length, :].to(self.device) # for padded prefix
                # # self.prefix_enc_outputs = self.linear(self.prefix_enc_outputs)
                # # # for value-attended enc output vs enc output cos sim
                # # prefix_enc_outputs = torch.cat([key_values_1[1][:,5,:,:,:].squeeze(1).reshape(batch_size, self.prefix_length, -1), self.enc_outputs], dim=1) # (batch_size, prefix_length + seq_length, input_size)
                # # self.prefix_enc_outputs = self.attention(prefix_enc_outputs.permute(1,0,2), prefix_enc_outputs.permute(1,0,2), prefix_enc_outputs.permute(1,0,2))[0].permute(1,0,2)[:, self.config.prefix_length:, :].to(self.device) # for padded prefix
                # # self.prefix_enc_outputs = self.linear(self.prefix_enc_outputs)
                # # for key-attended enc output vs enc output cos sim
                # prefix_enc_outputs_key = torch.cat([key_values_1[0][:,5,:,:,:].squeeze(1).reshape(batch_size, self.prefix_length, -1), self.enc_outputs], dim=1) # (batch_size, prefix_length + seq_length, input_size)
                # prefix_enc_outputs_val = torch.cat([key_values_1[1][:,5,:,:,:].squeeze(1).reshape(batch_size, self.prefix_length, -1), self.enc_outputs], dim=1) # (batch_size, prefix_length + seq_length, input_size)
                # self.prefix_enc_outputs = self.attention(self.enc_outputs.permute(1,0,2), prefix_enc_outputs_key.permute(1,0,2), prefix_enc_outputs_val.permute(1,0,2))[0].permute(1,0,2) #[:, self.config.prefix_length:, :].to(self.device) # for padded prefix
                # self.prefix_enc_outputs = self.linear(self.prefix_enc_outputs)
                
                # for enc output sub
                key_values_1_0 = key_values_1[0].clone()
                key_values_1_1 = key_values_1[1].clone()
                # for enc0 substitution
                # key_values_1_1[:,0,:,:,:] = self.enc_inputs.reshape(key_values_3.size()[0], self.prefix_length, self.n_heads, self.n_embd_per_head)
                # for enc5 substitution
                # key_values_1_0[:,5,:,:,:] = self.enc_outputs.reshape(key_values_3.size()[0], self.prefix_length, self.n_heads, self.n_embd_per_head)
                key_values_1_1[:,5,:,:,:] = self.enc_outputs.reshape(key_values_3.size()[0], self.prefix_length, self.n_heads, self.n_embd_per_head)
                key_values_1 = (key_values_1_0, key_values_1_1)

                # # self.prefix_enc_outputs = self.linear(key_values_1[1][:,5,:,:,:].squeeze(1).reshape(batch_size, self.prefix_length, -1)).to(self.device)
                # # self.prefix_enc_outputs_1 = self.linear_2(key_values_1[1][:,0,:,:,:].squeeze(1).reshape(batch_size, self.prefix_length, -1)).to(self.device)
                # # self.prefix_enc_outputs_2 = self.linear_1(key_values_1[1][:,5,:,:,:].squeeze(1).reshape(batch_size, self.prefix_length, -1)).to(self.device)

                key_values_2 = key_values_2.permute(0, 2, 1, 3, 4).split(self.n_layers, dim = 1)
                key_values_3 = key_values_3.permute(0, 2, 1, 3, 4).split(self.n_layers, dim = 1)
                
                prefix_dict['encoder_prefix'] = key_values_1
                prefix_dict['cross_prefix'] = key_values_2
                prefix_dict['decoder_prefix'] = key_values_3

                # for decoder output cos sim
                # dec_outputs = self.model.model(prefix_inputs, prefix=prefix_dict)
                # self.dec_outputs = dec_outputs.last_hidden_state.to(self.device)
                # prefix_dec_outputs = torch.cat([key_values_3[1][:,5,:,:,:].squeeze(1).reshape(batch_size, self.prefix_length, -1), self.dec_outputs], dim=1) # (batch_size, prefix_length + seq_length, input_size)
                # self.prefix_dec_outputs = self.attention(prefix_dec_outputs.permute(1,0,2), prefix_dec_outputs.permute(1,0,2), prefix_dec_outputs.permute(1,0,2))[0].permute(1,0,2)[:, :self.config.prefix_length, :].to(self.device) # for padded prefix
                # self.prefix_dec_outputs = self.linear(self.prefix_dec_outputs)

            elif self.config.prefix_type == "ptuning":
                prefix_inputs = self.prefix_ids.expand(len(prefix_inputs), -1) # for prefix tuning
                prefix_1 = self.wte_1(prefix_inputs) # (batch_size, prefix_length, input_size)
                prefix_2 = self.wte_2(prefix_inputs) # (batch_size, prefix_length, input_size)
                prefix_3 = self.wte_3(prefix_inputs) # (batch_size, prefix_length, input_size)

                embds_1 = prefix_1
                key_values_1 = self.control_trans_1(embds_1) #embds)  # batch_size x prefix_length x n_layers*2*input_size
                key_values_1 = key_values_1.view(
                    key_values_1.size()[0], self.prefix_length, self.n_layers * 2, self.n_heads, self.n_embd_per_head # original
                )  # *2 for key and value
                key_values_1 = self.dropout(key_values_1) # n_layers * (2 x batch_size x n_heads x prefix_length x n_embd_per_head)

                ##
                embds_2 = prefix_2
                key_values_2 = self.control_trans_2(embds_2) #embds)  # batch_size x prefix_length x n_layers*2*input_size
                key_values_2 = key_values_2.view(
                    key_values_2.size()[0], self.prefix_length, self.n_layers * 2, self.n_heads, self.n_embd_per_head # original
                )  # *2 for key and value
                key_values_2 = self.dropout(key_values_2) # n_layers * (2 x batch_size x n_heads x prefix_length x n_embd_per_head)

                ### 
                embds_3 = prefix_3
                key_values_3 = self.control_trans_3(embds_3) #embds)  # batch_size x prefix_length x n_layers*2*input_size
                key_values_3 = key_values_3.view(
                    key_values_3.size()[0], self.prefix_length, self.n_layers * 2, self.n_heads, self.n_embd_per_head # original
                )  # *2 for key and value
                key_values_3 = self.dropout(key_values_3) # n_layers * (2 x batch_size x n_heads x prefix_length x n_embd_per_head)

                prefix_dict = {} # need: (bz, n_encoder_layers, prefix_len, head_num, dim)

                # For regular (only enc) prefix
                key_values_1 = key_values_1.permute(0, 2, 1, 3, 4).split(self.n_layers, dim = 1)
                key_values_2 = key_values_2.permute(0, 2, 1, 3, 4).split(self.n_layers, dim = 1)
                key_values_3 = key_values_3.permute(0, 2, 1, 3, 4).split(self.n_layers, dim = 1)
                
                prefix_dict['encoder_prefix'] = key_values_1
                prefix_dict['cross_prefix'] = key_values_2
                prefix_dict['decoder_prefix'] = key_values_3
            
        # encoder inputs
        if self.config.use_enc_src_parse:
            input_texts = [f"{src_sent} {self.config.sep_token} {src_synt} {self.config.sep_token} {tgt_synt}" for src_sent, src_synt, tgt_synt in zip(src_sents, src_synts, tgt_synts)]
        else:
            # new
            input_texts = [f"{src_sent} {self.config.sep_token} {tgt_synt}" for src_sent, tgt_synt in zip(src_sents, tgt_synts)] 
            # input_texts = [f"{src_sent}" for src_sent in src_sents]
        inputs = self.tokenizer(input_texts, return_tensors='pt', padding=True)
        # print('inputs', type(inputs))

        enc_idxs = inputs['input_ids'] # (bsz, seq_len)
        enc_attn = inputs['attention_mask']
        #print("attention mask", enc_attn.size())
        # Attention mask should be of size (6, 1, 54, 54), but is torch.Size([6, 1, 24, 24]) when batch is 24
        
        if self.config.model_type == "prompt":
            enc_idxs = torch.cat([prefix, enc_idxs], dim = 1)
            # print('\n enc attn 1', enc_attn)
            enc_attn = self.resolve_attn_mask(enc_attn)
            # print('\n enc attn', enc_attn)
            prefix_dict = None
        
        enc_idxs = enc_idxs.cuda()
        enc_attn = enc_attn.cuda()
        
        if tgt_sents is None:
            return enc_idxs, enc_attn, None, None, None
        
        # decoder inputs
        if self.config.use_dec_tgt_parse:
            output_texts = [f"{tgt_synt} {self.config.sep_token} {tgt_sent}" for tgt_synt, tgt_sent in zip(tgt_synts, tgt_sents)]
        else:
            output_texts = tgt_sents
        outputs = self.tokenizer(output_texts, return_tensors='pt', padding=True)

        outputs = outputs.to(self.device)
        batch_size = enc_idxs.size(0)
        
        
        padding = torch.ones((batch_size, 1), dtype=torch.long, device = self.device)
        padding[:] = self.tokenizer.eos_token_id
        dec_idxs = torch.cat((padding, outputs['input_ids']), dim=1)
        dec_attn = torch.cat((torch.ones((batch_size, 1), dtype=torch.long, device = self.device), outputs['attention_mask']), dim=1)
            
        # labels
        padding = torch.ones((batch_size, 1), dtype=torch.long, device = self.device)
        padding[:] = self.tokenizer.pad_token_id
        raw_lbl_idxs = torch.cat((dec_idxs[:, 1:], padding), dim=1)
        lbl_attn = torch.cat((dec_attn[:, 1:], torch.zeros((batch_size, 1), dtype=torch.long, device = self.device)), dim=1)
        lbl_idxs = raw_lbl_idxs.masked_fill(lbl_attn==0, -100) # ignore padding
        
        # print('decoder', dec_idxs.size())
        dec_idxs = dec_idxs.cuda()
        dec_attn = dec_attn.cuda()
        lbl_idxs = lbl_idxs.cuda()
        
        if self.show_demo_examples:
            print()
            for i in range(3):
                print(f"IN:\n {input_texts[i]}")
                print(f"OUT:\n {output_texts[i]}")
            self.show_demo_examples = False
        
        return enc_idxs, enc_attn, dec_idxs, dec_attn, lbl_idxs, prefix_dict

    # def forward(self, src_sents, src_synts, tgt_synts, tgt_sents):
    def forward(self, enc_idxs, enc_attn, dec_idxs, dec_attn, lbl_idxs, prefix_dict):
        # enc_idxs, enc_attn, dec_idxs, dec_attn, lbl_idxs = self.process_data(src_sents, src_synts, tgt_synts, tgt_sents)
        # print(enc_idxs.size())
        # for cos sim loss

        outputs = self.model(input_ids=enc_idxs, 
                             prefix = prefix_dict,
                             attention_mask=enc_attn, 
                             decoder_input_ids=dec_idxs, 
                             decoder_attention_mask=dec_attn, 
                             labels=lbl_idxs, 
                             return_dict=True)
        
        if self.config.prefix_type == "attention0":
            # # sim1 = self.prefix_criterion(F.log_softmax(self.prefix_enc_outputs/ 1, dim=1), F.softmax(self.enc_outputs/ 1, dim=1))
            # # for enc input cos sim
            # # sim1 = torch.mean(1 - torch.abs(self.prefix_criterion(self.prefix_enc_inputs, self.enc_inputs))).to(outputs['loss'].device)
            # # for enc output cos sim
            # sim1 = torch.mean(1 - torch.abs(self.prefix_criterion(self.prefix_enc_outputs, self.enc_outputs))).to(outputs['loss'].device)
            # # for dec output cos sim
            # # sim1 = torch.mean(1 - torch.abs(self.prefix_criterion(self.prefix_dec_outputs, self.dec_outputs))).to(outputs['loss'].device)

            # # sim2 = torch.abs(torch.mean(self.prefix_criterion(self.prefix_enc_outputs_2, self.enc_outputs_2)))
            # # loss = outputs['loss'] + sim1.to(outputs['loss'].device) + sim2.to(outputs['loss'].device)
            loss = outputs['loss'] # + self.mu * sim1
            # loss = outputs['loss'] + sim1.to(outputs['loss'].device)
            
        else:
            loss = outputs['loss']

        return loss
    
    # def generate(self, src_sents, src_synts, tgt_synts, num_beams=4):
    def generate(self, enc_idxs, enc_attn, prefix_dict, num_beams=4):
        
        self.eval()
        
        max_length = self.config.max_tgt_synt_len + self.config.max_tgt_sent_len if self.config.use_dec_tgt_parse else self.config.max_tgt_sent_len
        
        # enc_idxs, enc_attn, _, _, _ = self.process_data(src_sents, src_synts, tgt_synts)
        with torch.no_grad():
            outputs = self.model.generate(input_ids=enc_idxs, 
                                          prefix=prefix_dict,
                                          attention_mask=enc_attn, 
                                          num_beams=num_beams, 
                                          max_length=max_length)
        
        final_outputs = []
        for output in outputs:
            final_output = self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if self.config.use_dec_tgt_parse:
                if self.config.sep_token in final_output:
                    final_output = final_output.split(self.config.sep_token, 1)[1]
            final_outputs.append(final_output.strip())
            
        self.train()
        
        return final_outputs
