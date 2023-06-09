import os, logging, pprint, json
import numpy as np
import stanza
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
import ipdb

# configuration
parser = ArgumentParser()
parser.add_argument('--data', default="./para-nmt-50m/para-nmt-50m.txt")
args = parser.parse_args()

# logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)
logger.info(f"\n{pprint.pformat(vars(args), indent=4)}")

parser = stanza.Pipeline(lang='en', processors='tokenize')


logger.info(f"loading {args.data}")
with open(args.data) as fp:
    lines = fp.readlines()

para_lens = []
for line in tqdm(lines):
    try:
        line = line.strip().split("\t")
        para = line[1]
        para_obj = parser(para)
        para_lens.append(len(para_obj.sentences[0].tokens))
    except KeyboardInterrupt:
        raise
    except:
        continue

print(f"Avg Para Len: {np.mean(para_lens)}")

