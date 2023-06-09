import os, logging, pprint, json
import numpy as np
import stanza
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
import ipdb

# configuration
parser = ArgumentParser()
parser.add_argument('--data', default="./parabank2/parabank2.tsv")
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
n_para = []
for line in tqdm(lines):
    try:
        line = line.strip().split("\t")
        paras = line[2:]
    except KeyboardInterrupt:
        raise
    except:
        continue
    
    n_para.append(len(paras))
    para_len = []
    for para in paras:
        try:
            para_obj = parser(para)
            para_len.append(len(para_obj.sentences[0].tokens))
        except KeyboardInterrupt:
            raise
        except:
            continue
    
    if len(para_len) > 0:
        para_lens.append(np.mean(para_len))
    

print(f"Avg Para Len: {np.mean(para_lens)}")
print(f"Avg #Para: {np.mean(n_para)}")

