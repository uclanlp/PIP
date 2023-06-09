import os, logging, pprint, json
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
import ipdb

# configuration
parser = ArgumentParser()
parser.add_argument('--bank2', default="./parabank2/parabank2.tsv")
parser.add_argument('--nmt', default="./para-nmt-50m/para-nmt-50m.txt")
parser.add_argument('--outbank2', default="./merge/parabank2_out1.txt")
parser.add_argument('--outnmt', default="./merge/paranmt_out1.txt")
args = parser.parse_args()

# logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)
logger.info(f"\n{pprint.pformat(vars(args), indent=4)}")


logger.info(f"loading {args.bank2}")
with open(args.bank2) as fp:
    lines1 = fp.readlines()

logger.info(f"loading {args.nmt}")
with open(args.nmt) as fp:
    lines2 = fp.readlines()
    
fp1 = open(args.outbank2, "w")
fp2 = open(args.outnmt, "w")
    
map_nmt = {}
for i, line2 in tqdm(enumerate(lines2), total=len(lines2)):
    line2 = line2.strip()
    s1, _ = line2.split("\t", 1)
    s1 = s1.replace(" ", "")
    map_nmt[s1] = i

n_data = 0
for i, line1 in tqdm(enumerate(lines1), total=len(lines1)):
    line1 = line1.strip()
    try:
        score, s1, _ = line1.split("\t", 2)
    except KeyboardInterrupt:
        raise
    except:
        continue
    s1 = s1.replace(" ", "")
    
    if s1 not in map_nmt:
        continue
    
    n_data += 1
    outnmt = lines2[map_nmt[s1]].strip()
    outbank2 = line1
    
    fp1.write(outbank2+"\n")
    fp2.write(outnmt+"\n")
    
print(f"{n_data/len(lines1)} ({n_data}/{len(lines1)}) parabank2")
print(f"{n_data/len(lines2)} ({n_data}/{len(lines2)}) paranmt")
        
fp1.close()
fp2.close()