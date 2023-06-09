import os, logging, pprint, json
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
import ipdb

# configuration
parser = ArgumentParser()
parser.add_argument('--amr', default="./merge/paraamr_out5.txt.parse_")
parser.add_argument('--out', default="./merge/paraamr_out5.txt.parse")
args = parser.parse_args()

# logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)
logger.info(f"\n{pprint.pformat(vars(args), indent=4)}")


logger.info(f"loading {args.amr}")
with open(args.amr) as fp:
    lines = fp.readlines()
    

data = []
for line in tqdm(lines):
    jobj = json.loads(line)
    
    paraphrases = []
    para_set = set()
    for para in jobj["paraphrases"]:
        if para["para_text"] in para_set:
            print("ok")
            continue
        paraphrases.append(para)
        para_set.add(para["para_text"])
        
    dt = {
        "text": jobj["text"], 
        "paraphrases": paraphrases
    }
    data.append(dt)
        
with open(args.out, "w") as fp:
    for dt in data:
        fp.write(json.dumps(dt)+"\n")
