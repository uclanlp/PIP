import os, logging, pprint, json
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from zss import simple_distance, Node
from nltk.tree import Tree
import ipdb

# configuration
parser = ArgumentParser()
parser.add_argument('--data', default="./merge/paranmt_out5.txt.parse")
args = parser.parse_args()

# logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)
logger.info(f"\n{pprint.pformat(vars(args), indent=4)}")

def build_tree(s):
    old_t = Tree.fromstring(s)
    new_t = Node("ROOT")
    
    def create_tree(curr_t, t, d):
        if d > 4:
            return
        new_t = Node(t.label())
        curr_t.addkid(new_t)
        for i in t:
            if isinstance(i, Tree):
                create_tree(new_t, i, d+1)
    create_tree(new_t, old_t, 0)
    return new_t


def strdist(a, b):
    if a == b:
        return 0
    else:
        return 1


def compute_tree_edit_distance(pred_parse, ref_parse):
    return simple_distance(build_tree(ref_parse), build_tree(pred_parse), label_dist=strdist)

logger.info(f"loading {args.data}")
with open(args.data) as fp:
    lines = fp.readlines()

    
N = 10000
scores1 = []
for i, line in tqdm(enumerate(lines[:N]), total=len(lines[:N])):
    jobj = json.loads(line)
    sc = []
    for p in jobj["paraphrases"]:
        try:
            sc_ = compute_tree_edit_distance(jobj["text_parse"], p["para_parse"])
            sc.append(sc_)
        except KeyboardInterrupt:
            raise
        except:
            continue
        
    # sc = [compute_tree_edit_distance(jobj["text_parse"], p["para_parse"]) for p in jobj["paraphrases"]]
    scores1.append(np.mean(sc))

scores1 = [s for s in scores1 if not np.isnan(s)]

logger.info(f"Metric TED: {np.mean(scores1)}")
    
