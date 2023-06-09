import os, logging, pprint, json
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
import ipdb

# configuration
parser = ArgumentParser()
parser.add_argument('--amr', default="./merge/paraamr_out3.txt.sim")
parser.add_argument('--flag', default="./merge/out3_flags.txt")
parser.add_argument('--output', default="./merge/paraamr_out4.txt.sim")
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
    
        jobj = json.loads(line)

        paraphrases = []
        for para in jobj["paraphrase"]:
            if para["text"].lower() == jobj["text"].lower():
                continue
            paraphrases.append(para)
        
        oobj = {"text": jobj["text"], "paraphrase": paraphrases}
        
        fp.write(json.dumps(oobj)+"\n")