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

# set GPU device
#torch.cuda.set_device(config.gpu_device)

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

def evaluate(model, eval_data, meteor_eval, rouge_eval, output_dir, config, mode, show=True):
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
            
            #print(enc_idxs.size())

            # forard model
            # loss = model(src_sents, src_synts, tgt_synts, tgt_sents)
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
        rs = rouge_eval.get_scores([eval_output], [eval_targ])
        rouge1.append(rs['rouge-1'][0]['f'][0])
        rouge2.append(rs['rouge-2'][0]['f'][0])
        rougel.append(rs['rouge-l'][0]['f'][0])
    
    eval_scores = {"loss": avg_loss, 
                   "bleu": run_multi_bleu(pred_file, gold_file),
                    "rouge1": np.mean(rouge1)*100.0, 
                   "rouge2": np.mean(rouge2)*100.0, 
                   "rougel": np.mean(rougel)*100.0} 
                   # "meteor": np.mean(meteor)*100.0, 

    
    if show:
        print("-------------------------------------------------------")
        print(f"{mode.capitalize()}")
        print("-------------------------------------------------------")
        print("LOSS:   {:6.3f}    BLEU:    {:5.2f}".format(eval_scores["loss"], eval_scores["bleu"]))
        # print("LOSS:   {:6.3f}    BLEU:    {:5.2f}    METEOR:  {:5.2f}".format(eval_scores["loss"], eval_scores["bleu"], eval_scores["meteor"]))
        print("ROUGE-1: {:5.2f}    ROUGE-2: {:5.2f}    ROUGE-L: {:5.2f}".format(eval_scores["rouge1"], eval_scores["rouge2"], eval_scores["rougel"]))
        print("-------------------------------------------------------")
    
    return eval_scores, eval_outputs
            
tokenizer = BartTokenizer.from_pretrained(config.pretrained_model, cache_dir=config.cache_dir)
tokenizer.add_tokens([config.sep_token])

# train_data = load_data(config.train_src_sent_file, config.train_src_synt_file, config.train_tgt_sent_file, config.train_tgt_synt_file, tokenizer, config)
dev_data = load_data(config.dev_src_sent_file, config.dev_src_synt_file, config.dev_tgt_sent_file, config.dev_tgt_synt_file, tokenizer, config)
pan_data = load_data(config.pan_src_sent_file, config.pan_src_synt_file, config.pan_tgt_sent_file, config.pan_tgt_synt_file, tokenizer, config)
mrpc_data = load_data(config.mrpc_src_sent_file, config.mrpc_src_synt_file, config.mrpc_tgt_sent_file, config.mrpc_tgt_synt_file, tokenizer, config)
quora_data = load_data(config.quora_src_sent_file, config.quora_src_synt_file, config.quora_tgt_sent_file, config.quora_tgt_synt_file, tokenizer, config)

# initialize distributed training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using %d GPUs for training" % config.gpu_num)
device_ids = [i for i in range(config.gpu_num)]

# initialize the model
if config.model_type == "seq2seq":
    model = ParaphraseModel(config, tokenizer, device).to(device)
elif config.model_type == "aesop":
    config.use_enc_src_parse = True
    config.use_dec_tgt_parse = True
    model = ParaphraseModel(config, tokenizer, device).to(device)

model = nn.DataParallel(model, device_ids)

if config.model_dir != None:
    logger.info("Loading checkpoint from %s" % config.model_dir)
    model.load_state_dict(torch.load(os.path.join(config.model_dir, "best_model.mdl")), strict=False)
#model.cuda(device=config.gpu_device)

# scorer
# meteor_eval = Meteor()
# rouge_eval = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
#                          max_n=2,
#                          limit_length=True,
#                          length_limit=100,
#                          length_limit_type='words',
#                          apply_avg=False,
#                          apply_best=False,
#                          alpha=0.5,  # Default F1_score
#                          weight_factor=1.2,
#                          stemming=True)
meteor_eval = None
rouge_eval = rouge.Rouge(metrics=['rouge-1', 'rouge-2', 'rouge-l'])


logger.info("Start inferencing ...")

model.eval()
dev_scores, dev_outputs = evaluate(model, dev_data, meteor_eval, rouge_eval, output_dir, config, "dev")
logger.info({"dev_loss": dev_scores})

dev_scores = dev_scores
# eval test set
pan_scores, pan_outputs = evaluate(model, pan_data, meteor_eval, rouge_eval, output_dir, config, "test_pan")
mrpc_scores, mrpc_outputs = evaluate(model, mrpc_data, meteor_eval, rouge_eval, output_dir, config, "test_mrpc")
quora_scores, quora_outputs = evaluate(model, quora_data, meteor_eval, rouge_eval, output_dir, config, "test_quora")
logger.info({"pan_scores": pan_scores})
logger.info({"mrpc_scores": mrpc_scores})
logger.info({"quora_scores": quora_scores})

logger.info(log_path)
logger.info("Done!")
