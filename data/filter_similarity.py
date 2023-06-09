import os, logging, pprint, json
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
import ipdb

# configuration
parser = ArgumentParser()
parser.add_argument('--inbank1', default="./merge/parabank1_out2.txt")
parser.add_argument('--inbank2', default="./merge/parabank2_out2.txt")
parser.add_argument('--innmt', default="./merge/paranmt_out2.txt")
parser.add_argument('--outbank1', default="./merge/parabank1_out3.txt")
parser.add_argument('--outbank2', default="./merge/parabank2_out3.txt")
parser.add_argument('--outnmt', default="./merge/paranmt_out3.txt")
parser.add_argument('--thres1', type=float, default=0.5)
parser.add_argument('--thres2', type=float, default=0.7)
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
    
n_data = 0
for line1, line2, line3 in tqdm(zip(lines1, lines2, lines3), total=len(lines1)):
    try:
        score1 = float(line2.split("\t", 1)[0])
        score2 = float(line3.strip().split("\t")[-1])
    except KeyboardInterrupt:
        raise
    except:
        continue
    
    if score1 < args.thres1 or score2 < args.thres2:
        continue
    
    n_data += 1
    fp1.write(line1)
    fp2.write(line2)
    fp3.write(line3)
    
print(f"{n_data/len(lines1)} ({n_data}/{len(lines1)})")
        
fp1.close()
fp2.close()
fp3.close()
