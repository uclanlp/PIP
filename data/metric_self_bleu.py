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
parser.add_argument('--num', type=int, default=5)
args = parser.parse_args()

# logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)
logger.info(f"\n{pprint.pformat(vars(args), indent=4)}")


logger.info(f"loading {args.data}")
with open(args.data) as fp:
    lines = fp.readlines()
    
scores1 = []
for i, line in tqdm(enumerate(lines), total=len(lines)):
    jobj = json.loads(line)
    
    paraphrases = jobj["paraphrases"]
    np.random.shuffle(paraphrases)
    while len(paraphrases) < args.num:
        paraphrases += paraphrases
    paraphrases = paraphrases[:args.num]
    
    sc = []
    for i in range(args.num):
        for j in range(args.num):
            if i==j:
                continue
            sc.append(sentence_bleu([paraphrases[i]["para_tokens"].split(" ")], paraphrases[j]["para_tokens"].split(" ")))
            
    scores1.append(np.mean(sc))

logger.info(f"Metric Self-BLEU: {100.0*np.mean(scores1)}")
    
