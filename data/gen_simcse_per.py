import os, logging, pprint, json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
import ipdb

# configuration
parser = ArgumentParser()
parser.add_argument('--data', default="./merge/paraamr_out5.txt.parse.per")
parser.add_argument('--output', default="./merge/paraamr_out5.per.simcse.csv")
parser.add_argument('--num', type=int, default=5)
parser.add_argument('--per_thres', type=float, default=30.0)
args = parser.parse_args()

# logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)
logger.info(f"\n{pprint.pformat(vars(args), indent=4)}")


logger.info(f"loading {args.data}")
with open(args.data) as fp:
    lines = fp.readlines()
jobjs = [json.loads(line) for line in lines]
    
data = []
for i, jobj in tqdm(enumerate(jobjs), total=len(jobjs)):
    
    np.random.shuffle(jobj["paraphrases"])
    paraphrases = [para for para in jobj["paraphrases"] if para["perplexity"] < args.per_thres]
    paraphrases = paraphrases[:args.num]
    
    for j, para in enumerate(paraphrases):
        data.append((jobj["text"], para["para_text"], jobjs[(i+1)%len(jobjs)]["text"]))
        
logger.info(f"#Instances: {len(data)}")

pd.DataFrame(data, columns =['sent0', 'sent1', 'hard_neg']).to_csv(args.output, index=False)
    
