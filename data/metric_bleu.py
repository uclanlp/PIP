import os, logging, pprint, json
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from nltk.translate.bleu_score import sentence_bleu
import ipdb

# configuration
parser = ArgumentParser()
parser.add_argument('--data', default="./merge/paranmt_out5.txt.parse")
args = parser.parse_args()

# logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)
logger.info(f"\n{pprint.pformat(vars(args), indent=4)}")


logger.info(f"loading {args.data}")
with open(args.data) as fp:
    lines = fp.readlines()
    
# with open("./merge/out4_flags.txt") as fp:
#     flags = fp.readlines()
    
scores1 = []
for i, line in tqdm(enumerate(lines), total=len(lines)):
    jobj = json.loads(line)
    
    sc = [sentence_bleu([jobj["text_tokens"].split(" ")], p["para_tokens"].split(" ")) for p in jobj["paraphrases"]]
#     if flags[i].strip() == "1":
    scores1.append(np.mean(sc))

logger.info(f"Metric 1-BLEU: {100.0*(1.0-np.mean(scores1))}")
    
