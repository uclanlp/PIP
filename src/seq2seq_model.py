import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BartForConditionalGeneration, BartConfig
from transformers import BartForConditionalGeneration, BartModel, GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
from transformers.models.bart.modeling_bart import BartEncoder, BaseModelOutput, _expand_mask
# from prefix2 import PrefixGenBartForConditionalGeneration
import ipdb

class ParaphraseModel(nn.Module):
    def __init__(self, config, tokenizer, device, debug=True):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        if config.pretrained_model == "facebook/bart-base":
            self.model = BartForConditionalGeneration.from_pretrained(self.config.pretrained_model, cache_dir=self.config.cache_dir)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.debug = debug
        self.device = device
        if self.debug:
            self.show_demo_examples = True
        self.criterion = nn.CrossEntropyLoss()
        
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
    
    # def forward(self, src_sents, src_synts, tgt_synts, tgt_sents):
    def forward(self, enc_idxs, enc_attn, dec_idxs, dec_attn, lbl_idxs):
        # enc_idxs, enc_attn, dec_idxs, dec_attn, lbl_idxs = self.process_data(src_sents, src_synts, tgt_synts, tgt_sents)
        # print(enc_idxs.size())
        # print('\n labels', enc_idxs.size(), lbl_idxs.size())
        if self.config.pretrained_model == "facebook/bart-base":
            outputs = self.model(input_ids=enc_idxs, 
                                attention_mask=enc_attn, 
                                decoder_input_ids=dec_idxs, 
                                decoder_attention_mask=dec_attn, 
                                labels=lbl_idxs, 
                                return_dict=True)
        elif self.config.pretrained_model == "gpt2":
            outputs = self.model(input_ids=dec_idxs,
                                encoder_hidden_states=enc_idxs,
                                labels=dec_idxs, 
                                return_dict=True)

        loss = outputs['loss']
        
        return loss
    
    # def generate(self, src_sents, src_synts, tgt_synts, num_beams=4):
    def generate(self, enc_idxs, enc_attn, num_beams=4):
        
        self.eval()
        
        max_length = self.config.max_tgt_synt_len + self.config.max_tgt_sent_len if self.config.use_dec_tgt_parse else self.config.max_tgt_sent_len
        
        # enc_idxs, enc_attn, _, _, _ = self.process_data(src_sents, src_synts, tgt_synts)
        with torch.no_grad():
            if self.config.pretrained_model == "facebook/bart-base":
                outputs = self.model.generate(input_ids=enc_idxs, 
                                            attention_mask=enc_attn, 
                                            num_beams=num_beams, 
                                            max_length=max_length)
            elif self.config.pretrained_model == "gpt2":
                outputs = self.model.generate(input_ids=enc_idxs,
                                            labels=enc_idxs,  
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
