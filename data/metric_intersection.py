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
    
# with open("./merge/out4_flags.txt") as fp:
#     flags = fp.readlines()
    
scores1 = []
for i, line in tqdm(enumerate(lines), total=len(lines)):
    jobj = json.loads(line)
    
    text_set = set(jobj["text_tokens"].split(" "))
    para_sets = [set(p["para_tokens"].split(" ")) for p in jobj["paraphrases"]]
    sc = [1.0*len(text_set & para_set)/len(text_set | para_set) for para_set in para_sets]
    scores1.append(np.mean(sc))

logger.info(f"Metric Intersection/Union: {100.0*np.mean(scores1)}")
    
