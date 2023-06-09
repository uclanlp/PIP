import os, logging, pprint, json
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
import ipdb

# configuration
parser = ArgumentParser()
parser.add_argument('--data', default="./merge/paranmt_out5.txt.sim")
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
    
scores = []
for i, line in tqdm(enumerate(lines), total=len(lines)):
    jobj = json.loads(line)
    sc = [p["similarity"] for p in jobj["paraphrases"]]
#     if flags[i].strip() == "1":
    scores.append(np.mean(sc))

logger.info(f"Metric SimCSE: {100.0*np.mean(scores)}")
    
