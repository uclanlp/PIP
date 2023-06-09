import os, logging, pprint, json
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
import ipdb

# configuration
parser = ArgumentParser()
parser.add_argument('--innmt', default="./merge/paranmt_out4.txt.parse")
parser.add_argument('--inbank1', default="./merge/parabank1_out4.txt.parse")
parser.add_argument('--inbank2', default="./merge/parabank2_out4.txt.parse")
parser.add_argument('--inamr', default="./merge/paraamr_out4.txt.parse")
parser.add_argument('--flag', default="./merge/out4_brok_flags.txt")
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

    

flags = []
for line1, line2, line3, line4 in zip(lines1, lines2, lines3, lines4):
    if "paraphrases" not in line1 or "paraphrases" not in line2 or "paraphrases" not in line3 or "paraphrases" not in line4:
        flags.append(0)
    else:
        flags.append(1)
        
logger.info(f"{sum(flags)/len(lines1)} ({sum(flags)}/{len(lines1)})")

with open(args.flag, "w") as fp:
    for flag in flags:
        fp.write(f"{flag}\n")

