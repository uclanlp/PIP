import os, sys, json, logging, time, pprint, _jsonnet, subprocess, random
import numpy as np
import torch
import rouge
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import BartTokenizer, AdamW, get_linear_schedule_with_warmup
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
from model import ParaphraseModel #, SynPG
from utils import load_data, load_special_tokens
# from evaluation.eval_utils import Meteor
import ipdb

# configuration
parser = ArgumentParser()
parser.add_argument('-c', '--config', required=True)
parser.add_argument('--seed', type=int, required=False)
args = parser.parse_args()

config = json.loads(_jsonnet.evaluate_file(args.config))
config = Namespace(**config)

if args.seed is not None:
    config.seed = args.seed

# fix random seed
random.seed(0)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.backends.cudnn.enabled = False

# logger and summarizer
timestamp = time.strftime('%Y%m%d_%H%M%S%f', time.localtime())[:-3]
output_dir = os.path.join(config.output_dir, timestamp)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
log_path = os.path.join(output_dir, "train.log")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]', 
                    handlers=[logging.FileHandler(os.path.join(output_dir, "train.log")), logging.StreamHandler()])
logger = logging.getLogger(__name__)
logger.info(f"\n{pprint.pformat(vars(config), indent=4)}")

# output
with open(os.path.join(output_dir, "config.json"), 'w') as fp:
    json.dump(vars(config), fp, indent=4)
    
def run_multi_bleu(input_file, reference_file):
    MULTI_BLEU_PERL = 'src/evaluation/apps/multi-bleu.perl'
    bleu_output = subprocess.check_output(
        "./{} -lc {} < {}".format(MULTI_BLEU_PERL, reference_file, input_file),
        stderr=subprocess.STDOUT, shell=True).decode('utf-8')
    bleu = float(bleu_output.strip().split("\n")[-1].split(",")[0].split("=")[1][1:])
    return bleu

def evaluate(epoch, model, eval_data, meteor_eval, rouge_eval, output_dir, config, mode, show=True):
    model.eval()
    avg_loss = 0.0
    eval_outputs = []
    with torch.no_grad():
        eval_loader = DataLoader(np.arange(len(eval_data)), batch_size=config.eval_batch_size, shuffle=False)
        for bid, eval_idxs in tqdm(enumerate(eval_loader), total=len(eval_loader), ncols=100, ascii=True):

            src_sents = [eval_data[i]["src_sent"] for i in eval_idxs]
            src_synts = [eval_data[i]["src_synt"] for i in eval_idxs]
            tgt_sents = [eval_data[i]["tgt_sent"] for i in eval_idxs]
            tgt_synts = [eval_data[i]["tgt_synt"] for i in eval_idxs]
            
            enc_idxs, enc_attn, dec_idxs, dec_attn, lbl_idxs = model.module.process_data(src_sents, src_synts, tgt_synts, tgt_sents)

            enc_idxs = enc_idxs.to(device)
            enc_attn = enc_attn.to(device)
            dec_idxs = dec_idxs.to(device)
            dec_attn = dec_attn.to(device)
            lbl_idxs = lbl_idxs.to(device)

            # forward model
            loss = model(enc_idxs, enc_attn, dec_idxs, dec_attn, lbl_idxs).sum()
            avg_loss += loss.item()

            outputs = model.module.generate(enc_idxs, enc_attn)
            print('outputs', outputs[0])
            eval_outputs.extend(outputs)
            
    avg_loss /= len(eval_loader)
            
    eval_targs = [dt["tgt_sent"] for dt in eval_data]
    
    pred_file = os.path.join(output_dir, f"{mode}_pred.txt")
    with open(pred_file, "w") as fp:
        for eval_output in eval_outputs:
            fp.write(eval_output+"\n")
    gold_file = os.path.join(output_dir, f"{mode}_gold.txt")
    with open(gold_file, "w") as fp:
        for eval_targ in eval_targs:
            fp.write(eval_targ+"\n")
            
    rouge1, rouge2, rougel = [], [], []
    for eval_targ, eval_output in zip(eval_targs, eval_outputs):
        if len(eval_output) >= 1:
            rs = rouge_eval.get_scores([eval_output], [eval_targ])
            # print(rs[0]['rouge-1'].keys())
            rouge1.append(rs[0]['rouge-1']['f'])
            rouge2.append(rs[0]['rouge-2']['f'])
            rougel.append(rs[0]['rouge-l']['f'])

    eval_scores = {"loss": avg_loss, 
                   "bleu": run_multi_bleu(pred_file, gold_file), 
                   "rouge1": np.mean(rouge1)*100.0, 
                   "rouge2": np.mean(rouge2)*100.0, 
                   "rougel": np.mean(rougel)*100.0}
                   # "meteor": np.mean(meteor)*100.0, 
                   
    if show:
        print("-------------------------------------------------------")
        print(f"Epoch {epoch} {mode.capitalize()}")
        print("-------------------------------------------------------")
        print("LOSS:   {:6.3f}    BLEU:    {:5.2f}".format(eval_scores["loss"], eval_scores["bleu"]))
        # print("LOSS:   {:6.3f}    BLEU:    {:5.2f}    METEOR:  {:5.2f}".format(eval_scores["loss"], eval_scores["bleu"], eval_scores["meteor"]))
        print("ROUGE-1: {:5.2f}    ROUGE-2: {:5.2f}    ROUGE-L: {:5.2f}".format(eval_scores["rouge1"], eval_scores["rouge2"], eval_scores["rougel"]))
        print("-------------------------------------------------------")
    
    return eval_scores, eval_outputs
            
tokenizer = BartTokenizer.from_pretrained(config.pretrained_model, cache_dir=config.cache_dir)
tokenizer.add_tokens([config.sep_token])

train_data = load_data(config.train_src_sent_file, config.train_src_synt_file, config.train_tgt_sent_file, config.train_tgt_synt_file, tokenizer, config)
dev_data = load_data(config.dev_src_sent_file, config.dev_src_synt_file, config.dev_tgt_sent_file, config.dev_tgt_synt_file, tokenizer, config)
pan_data = load_data(config.pan_src_sent_file, config.pan_src_synt_file, config.pan_tgt_sent_file, config.pan_tgt_synt_file, tokenizer, config)
mrpc_data = load_data(config.mrpc_src_sent_file, config.mrpc_src_synt_file, config.mrpc_tgt_sent_file, config.mrpc_tgt_synt_file, tokenizer, config)
quora_data = load_data(config.quora_src_sent_file, config.quora_src_synt_file, config.quora_tgt_sent_file, config.quora_tgt_synt_file, tokenizer, config)

# initialize distributed training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using %d GPUs for training" % config.gpu_num)
device_ids = [i for i in range(config.gpu_num)]

# initialize the model
model = ParaphraseModel(config, tokenizer, device).to(device)

model = nn.DataParallel(model, device_ids)

# optimizer
param_groups = [{'params': model.parameters(), 'lr': config.learning_rate, 'weight_decay': config.weight_decay}]
optimizer = AdamW(params=param_groups)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=(len(train_data)//config.train_batch_size+(len(train_data)%config.train_batch_size!=0)) * config.warmup_epoch,
                                            num_training_steps=(len(train_data)//config.train_batch_size+(len(train_data)%config.train_batch_size!=0)) * config.max_epoch)

meteor_eval = None
rouge_eval = rouge.Rouge(metrics=['rouge-1', 'rouge-2', 'rouge-l'])

# start training
logger.info("Start training ...")

logger.info("** {:.2f}M parameters **".format(sum(p.numel() for p in model.parameters())/1000000.0))
logger.info("** {:.2f}M ({:.2f}K) learnable parameters **".format(sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000.0, 
                                                        sum(p.numel() for p in model.parameters() if p.requires_grad)/1000.0))

best_dev_scores = {"loss": np.inf, "bleu": 0.0}
best_dev_epoch = -1
for epoch in range(1, config.max_epoch+1):
    logger.info(log_path)
    logger.info(f"Epoch {epoch}")
    
    train_loader = DataLoader(np.arange(len(train_data)), batch_size=config.train_batch_size, shuffle=True)
    
    model.train()
    
    avg_loss = 0.0
    
    for bid, train_idxs in tqdm(enumerate(train_loader), total=len(train_loader), ncols=100, ascii=True):

        src_sents = [train_data[i]["src_sent"] for i in train_idxs]
        src_synts = [train_data[i]["src_synt"] for i in train_idxs]
        tgt_sents = [train_data[i]["tgt_sent"] for i in train_idxs]
        tgt_synts = [train_data[i]["tgt_synt"] for i in train_idxs]

        enc_idxs, enc_attn, dec_idxs, dec_attn, lbl_idxs = model.module.process_data(src_sents, src_synts, tgt_synts, tgt_sents)

        enc_idxs = enc_idxs.to(device)
        enc_attn = enc_attn.to(device)
        dec_idxs = dec_idxs.to(device)
        dec_attn = dec_attn.to(device)
        lbl_idxs = lbl_idxs.to(device)
    
        # forward model
        loss = model(enc_idxs, enc_attn, dec_idxs, dec_attn, lbl_idxs).sum()
        avg_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clipping)
        optimizer.step()
        scheduler.step()
        
    avg_loss /= len(train_loader)
    print(f"Loss: {avg_loss}")
    
    # eval dev set
    model.eval()
    dev_scores, dev_outputs = evaluate(epoch, model, dev_data, meteor_eval, rouge_eval, output_dir, config, "dev")
    logger.info({"epoch": epoch, "dev_loss": dev_scores})
    
    if dev_scores["bleu"] > best_dev_scores["bleu"]:
        logger.info(f"Saving best model to {os.path.join(output_dir, 'best_model.mdl')}")
        torch.save(model.state_dict(), os.path.join(output_dir, "best_model.mdl"))
        best_dev_scores = dev_scores
        best_dev_epoch = epoch
        
        # eval test set
        pan_scores, pan_outputs = evaluate(epoch, model, pan_data, meteor_eval, rouge_eval, output_dir, config, "test_pan")
        mrpc_scores, mrpc_outputs = evaluate(epoch, model, mrpc_data, meteor_eval, rouge_eval, output_dir, config, "test_mrpc")
        quora_scores, quora_outputs = evaluate(epoch, model, quora_data, meteor_eval, rouge_eval, output_dir, config, "test_quora")
        logger.info({"epoch": epoch, "pan_scores": pan_scores})
        logger.info({"epoch": epoch, "mrpc_scores": mrpc_scores})
        logger.info({"epoch": epoch, "quora_scores": quora_scores})
    
    logger.info("Current best")
    logger.info({"best_epoch": best_dev_epoch, "best_scores": best_dev_scores})
        
logger.info(log_path)
logger.info("Done!")
