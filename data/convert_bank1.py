import os, logging, pprint, json
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
import ipdb

# configuration
parser = ArgumentParser()
parser.add_argument('--amr', default="./merge/parabank1_out3.txt")
parser.add_argument('--flag', default="./merge/out3_flags.txt")
parser.add_argument('--output', default="./merge/parabank1_out4.txt")
args = parser.parse_args()

# logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)
logger.info(f"\n{pprint.pformat(vars(args), indent=4)}")


logger.info(f"loading {args.amr}")
with open(args.amr) as fp:
    lines = fp.readlines()
    
with open(args.flag) as fp:
    flags = fp.readlines()

with open(args.output, "w") as fp:
    for flag, line in zip(flags, lines):
        if flag.strip() == "0":
            continue
        
        line = line.strip()
        line = line.split("\t")
        
        ref = line[0]
        pars = line[1:]
        
        dt = {
            "text": ref, 
            "paraphrase": pars
        }
        fp.write(json.dumps(dt)+"\n")
