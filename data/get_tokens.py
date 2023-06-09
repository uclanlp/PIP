import os, logging, pprint, json
import stanza
import torch
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
import ipdb

# configuration
parser = ArgumentParser()
parser.add_argument('--data', default="./merge/paranmt_out3.txt")
args = parser.parse_args()

# logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)
logger.info(f"\n{pprint.pformat(vars(args), indent=4)}")

parser = stanza.Pipeline(lang='en', processors='tokenize')

logger.info(f"loading {args.data}")
with open(args.data) as fp:
    lines = fp.readlines()
    
data = []
for line in tqdm(lines):
    jobj = json.loads(line)
    text_obj = parser(jobj["text"])
    try:
        text_tokens = " ".join([t.text for t in text_obj.sentences[0].tokens])
    except:
        text_tokens = jobj["text"]
    
    paraphrase = []
    for para in jobj["paraphrases"]:
        try:
            para_obj = parser(para)
            para_tokens = " ".join([t.text for t in para_obj.sentences[0].tokens])
        except:
            para_tokens = para
        
        paraphrase.append({"para_text": para, "para_tokens": para_tokens})
        
    dt = {
        "text": jobj["text"], 
        "text_tokens": text_tokens, 
        "paraphrases": paraphrase
    }
    data.append(dt)
    
with open(args.data+".token", "w") as fp:
    for dt in data:
        fp.write(json.dumps(dt)+"\n")
