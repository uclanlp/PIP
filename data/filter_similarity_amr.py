import os, logging, pprint, json
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
import ipdb

# configuration
parser = ArgumentParser()
parser.add_argument('--amr', default="./merge/paraamr_out4.txt.sim")
parser.add_argument('--output', default="./merge/out4_flags.txt")
parser.add_argument('--amr2', default="./merge/paraamr_out4.txt.sim2")
parser.add_argument('--thres', type=float, default=0.8)
args = parser.parse_args()

# logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)
logger.info(f"\n{pprint.pformat(vars(args), indent=4)}")


logger.info(f"loading {args.amr}")
with open(args.amr) as fp:
    lines = fp.readlines()
    

flags = []
data = []
for line in tqdm(lines):
    jobj = json.loads(line)
    
    paraphrases = []
    for para in jobj["paraphrase"]:
        if para["similarity"] < args.thres:
            continue
        paraphrases.append(para)
        
    if len(paraphrases) == 0:
        flags.append(0)
    else:
        flags.append(1)
        
    dt = {
        "text": jobj["text"], 
        "paraphrase": paraphrases
    }
    
    data.append(dt)
        
logger.info(f"{sum(flags)/len(lines)} ({sum(flags)}/{len(lines)})")

with open(args.output, "w") as fp:
    for flag in flags:
        fp.write(f"{flag}\n")

with open(args.amr2, "w") as fp:
    for dt in data:
        fp.write(json.dumps(dt)+"\n")
