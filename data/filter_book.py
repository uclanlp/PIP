import os, logging, pprint, json
import evaluate_seq2seq
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
import ipdb

# configuration
parser = ArgumentParser()
parser.add_argument('--input', default="./merge/paraamr_out5.txt.parse")
args = parser.parse_args()

# logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)
logger.info(f"\n{pprint.pformat(vars(args), indent=4)}")


logger.info(f"loading {args.input}")
with open(args.input) as fp:
    lines = fp.readlines()

output_fn = f"{args.input}.fil"
with open(output_fn, "w") as fp:
    n_data = 0
    for line in tqdm(lines, ncols=100, ascii=True):
        jobj = json.loads(line)

        if "paraphrases" in jobj:
            n_data += 1
            fp.write(line)
            
    logger.info(f"load {n_data}/{len(lines)}")
    
