import os, logging, pprint
import stanza
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
import ipdb

# configuration
parser = ArgumentParser()
parser.add_argument('--data_dir', required=True)
args = parser.parse_args()

# logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)
logger.info(f"\n{pprint.pformat(vars(args), indent=4)}")

files = [("train/src.txt", "train/src.parse"), 
         ("train/tgt.txt", "train/tgt.parse"),
         ("val/src.txt", "val/src.parse"),
         ("val/tgt.txt", "val/tgt.parse"),
         ("val/ref.txt", "val/ref.parse"),
         ("test/src.txt", "test/src.parse"),
         ("test/tgt.txt", "test/tgt.parse"),
         ("test/ref.txt", "test/ref.parse"),
        ]

parser = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')

def tree2str(tree):
    if tree.is_leaf():
        return []
    tokens = ["("]
    tokens.append(tree.label)
    for child in tree.children:
        tokens.extend(tree2str(child))
        
    tokens.append(")")
    return tokens

parse_label_set = set()

for text_fn, parse_fn in files:
    try:
        logger.info(f"Processing {os.path.join(args.data_dir, text_fn)}")
        with open(os.path.join(args.data_dir, text_fn)) as fp:
            lines = fp.readlines()
    except:
        logger.info(f"{os.path.join(args.data_dir, text_fn)} does not exist.")
        continue
    
    parses = []
    for line in tqdm(lines, ascii=True):
        text = line.strip()
        obj = parser(text)
        tree = obj.sentences[0].constituency
        # parse_tokens = tree2str(tree)
        # parse_label_set |= set(parse_tokens)
        # parses.append(" ".join(parse_tokens))
        parses.append(str(obj.sentences[0].constituency))
    
    logger.info(f"Saving {os.path.join(args.data_dir, parse_fn)}")
    with open(os.path.join(args.data_dir, parse_fn), "w") as fp:
        for parse in parses:
            fp.write(parse+"\n")

# logger.info(f"Saving {os.path.join(args.data_dir, 'parse_tokens.txt')}")
# with open(os.path.join(args.data_dir, "parse_tokens.txt"), "w") as fp:
#     for label in sorted(parse_label_set):
#         fp.write(label+"\n")
        
