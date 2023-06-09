import os, logging, pprint, json
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
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
    
scores1 = []
scores2 = []
for i, line in tqdm(enumerate(lines), total=len(lines)):
    jobj = json.loads(line)
    
    sc1 = [len(p["para_tokens"].split(" ")) for p in jobj["paraphrases"]]
    scores1.append(np.mean(sc1))
    
    sc2 = len(jobj["paraphrases"])
    scores2.append(sc2)

logger.info(f"Metric #words: {np.mean(scores1)}")
logger.info(f"Metric #paraphrases: {np.mean(scores2)}")
logger.info(f"Metric #paraphrases (max): {np.max(scores2)}")
    
