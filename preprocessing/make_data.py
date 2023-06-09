import os, logging, pprint, json
import stanza
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from nltk import ParentedTree
import ipdb

# configuration
parser = ArgumentParser()
parser.add_argument('--data_dir', required=True)
args = parser.parse_args()

# logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)
logger.info(f"\n{pprint.pformat(vars(args), indent=4)}")

# files = [("train/src-small.pseudo.parse", "train/src-small.parse", "train/p-src-small.txt", "train/p-src-small.parse", "train/p-tgt-small.txt", "train/p-tgt-small.parse")]
# files = [("train/src-medium.pseudo.parse", "train/src-medium.parse", "train/p-src-medium.txt", "train/p-src-medium.parse", "train/p-tgt-medium.txt", "train/p-tgt-medium.parse")]
files = [("train/src.pseudo.parse", "train/src.parse", "train/p-src-full.txt", "train/p-src-full.parse", "train/p-tgt-full.txt", "train/p-tgt-full.parse")]

for psd_prs_fn, ori_prs_fn, psrc_txt_fn, psrc_prs_fn, ptgt_txt_fn, ptgt_prs_fn in files:
    logger.info(f"Processing {os.path.join(args.data_dir, psd_prs_fn)}, {os.path.join(args.data_dir, ori_prs_fn)}")
    with open(os.path.join(args.data_dir, psd_prs_fn)) as fp:
        lines = fp.readlines()
    with open(os.path.join(args.data_dir, ori_prs_fn)) as fp:
        parses = fp.readlines()
        
    assert len(lines) == len(parses)
    
    psrc_txts = []
    psrc_prss = []
    ptgt_txts = []
    ptgt_prss = []
    for line, parse in tqdm(zip(lines, parses), total=len(lines), ascii=True):
        obj = json.loads(line)
        parse = parse.strip()
        
        for psd_txt, psd_prs in obj["pseudo"]:
            try:
                tree = ParentedTree.fromstring(psd_prs)
                psrc_txts.append(obj["text"])
                psrc_prss.append(parse)
                ptgt_txts.append(psd_txt)
                ptgt_prss.append(psd_prs)
            except:
                continue
            
            
    with open(os.path.join(args.data_dir, psrc_txt_fn), "w") as fp:
        for l in psrc_txts:
            fp.write(l+"\n")
    with open(os.path.join(args.data_dir, psrc_prs_fn), "w") as fp:
        for l in psrc_prss:
            fp.write(l+"\n")
    with open(os.path.join(args.data_dir, ptgt_txt_fn), "w") as fp:
        for l in ptgt_txts:
            fp.write(l+"\n")
    with open(os.path.join(args.data_dir, ptgt_prs_fn), "w") as fp:
        for l in ptgt_prss:
            fp.write(l+"\n")

