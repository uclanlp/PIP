import os, logging, pprint, json
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
import ipdb

# configuration
parser = ArgumentParser()
parser.add_argument('--innmt', default="./merge/paranmt_out4.txt")
parser.add_argument('--inbank1', default="./merge/parabank1_out4.txt")
parser.add_argument('--inbank2', default="./merge/parabank2_out4.txt")
parser.add_argument('--inamr', default="./merge/paraamr_out4.txt")
parser.add_argument('--outnmt', default="./merge/paranmt_out5.txt")
parser.add_argument('--outbank1', default="./merge/parabank1_out5.txt")
parser.add_argument('--outbank2', default="./merge/parabank2_out5.txt")
parser.add_argument('--outamr', default="./merge/paraamr_out5.txt")
args = parser.parse_args()

# logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)
logger.info(f"\n{pprint.pformat(vars(args), indent=4)}")


logger.info(f"loading {args.innmt}")
with open(args.innmt) as fp:
    lines1 = fp.readlines()
    
logger.info(f"loading {args.inbank1}")
with open(args.inbank1) as fp:
    lines2 = fp.readlines()
    
logger.info(f"loading {args.inbank2}")
with open(args.inbank2) as fp:
    lines3 = fp.readlines()
    
logger.info(f"loading {args.inamr}")
with open(args.inamr) as fp:
    lines4 = fp.readlines()
    
with open("./merge/out4_flags.txt") as fp:
    flags1 = fp.readlines()
    
with open("./merge/out4_brok_flags.txt") as fp:
    flags2 = fp.readlines()

    

fp1 = open(args.outnmt, "w")
fp2 = open(args.outbank1, "w")
fp3 = open(args.outbank2, "w")
fp4 = open(args.outamr, "w")

n_data = 0
for line1, line2, line3, line4, flag1, flag2 in zip(lines1, lines2, lines3, lines4, flags1, flags2):
    if flag1.strip() == "1" and flag2.strip() == "1":
        n_data += 1
        fp1.write(line1)
        fp2.write(line2)
        fp3.write(line3)
        fp4.write(line4)
        
logger.info(f"{n_data/len(lines1)} ({n_data}/{len(lines1)})")

fp1.close()
fp2.close()
fp3.close()
fp4.close()
