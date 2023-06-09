import os, logging, pprint, json
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
import ipdb

# configuration
parser = ArgumentParser()
parser.add_argument('--inbank1', default="./full/parabank1_out2.txt")
parser.add_argument('--inbank2', default="./full/parabank2_out2.txt")
parser.add_argument('--innmt', default="./full/paranmt_out2.txt")
parser.add_argument('--outbank1', default="./full/parabank1_out3.txt")
parser.add_argument('--outbank2', default="./full/parabank2_out3.txt")
parser.add_argument('--outnmt', default="./full/paranmt_out3.txt")
parser.add_argument('--num', type=int, default=400000)
args = parser.parse_args()

# logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)
logger.info(f"\n{pprint.pformat(vars(args), indent=4)}")


logger.info(f"loading {args.inbank1}")
with open(args.inbank1) as fp:
    lines1 = fp.readlines()

logger.info(f"loading {args.inbank2}")
with open(args.inbank2) as fp:
    lines2 = fp.readlines()

logger.info(f"loading {args.innmt}")
with open(args.innmt) as fp:
    lines3 = fp.readlines()
    
fp1 = open(args.outbank1, "w")
fp2 = open(args.outbank2, "w")
fp3 = open(args.outnmt, "w")

idxs = np.arange(len(lines1))
np.random.shuffle(idxs)

for i in tqdm(idxs[:args.num]):
    line1 = lines1[i].lower().strip().split("\t")
    text1 = line1[0]
    pars1 = line1[1:]
    dt1 = {
        "text": text1, 
        "paraphrases": pars1
    }
    fp1.write(json.dumps(dt1)+"\n")
    
    line2 = lines2[i].lower().strip().split("\t")
    text2 = line2[1]
    pars2 = line2[2:]
    dt2 = {
        "text": text2, 
        "paraphrases": pars2
    }
    fp2.write(json.dumps(dt2)+"\n")
    
    line3 = lines3[i].lower().strip().split("\t")
    text3 = line3[0]
    pars3 = line3[1]
    dt3 = {
        "text": text3, 
        "paraphrases": [pars3]
    }
    fp3.write(json.dumps(dt3)+"\n")
    
fp1.close()
fp2.close()
fp3.close()
