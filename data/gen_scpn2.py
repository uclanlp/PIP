import os, logging, pprint, json
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from nltk import ParentedTree
import ipdb

# configuration
parser = ArgumentParser()
parser.add_argument('--data', default="./merge/paranmt_out5.txt.parse")
parser.add_argument('--output', default="./merge/paranmt_out5.scpn")
parser.add_argument('--num', type=int, default=5)
args = parser.parse_args()

# logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)
logger.info(f"\n{pprint.pformat(vars(args), indent=4)}")


logger.info(f"loading {args.data}")
with open(args.data) as fp:
    lines = fp.readlines()
jobjs = [json.loads(line) for line in lines]

train_data = []
for i, jobj in tqdm(enumerate(jobjs[:-5000]), total=len(jobjs[:-5000])):
    
    try:
        ParentedTree.fromstring(jobj["text_parse"])
    except KeyboardInterrupt:
        raise
    except:
        continue
    
    np.random.shuffle(jobj["paraphrases"])
    paraphrases = jobj["paraphrases"][:args.num]
    
    train_data.append((jobj["text_tokens"], jobj["text_parse"], jobj["text_tokens"], jobj["text_parse"]))
    
    for j, para in enumerate(paraphrases):
        try:
            ParentedTree.fromstring(para["para_parse"])
        except KeyboardInterrupt:
            raise
        except:
            continue
        train_data.append((jobj["text_tokens"], jobj["text_parse"], para["para_tokens"], para["para_parse"]))
    
logger.info(f"#Train Instances: {len(train_data)}")

dev_data = []
for i, jobj in tqdm(enumerate(jobjs[-5000:]), total=len(jobjs[-5000:])):
    
    try:
        ParentedTree.fromstring(jobj["text_parse"])
    except KeyboardInterrupt:
        raise
    except:
        continue
    
    np.random.shuffle(jobj["paraphrases"])
    paraphrases = jobj["paraphrases"][:args.num]
    
    for j, para in enumerate(paraphrases):
        try:
            ParentedTree.fromstring(para["para_parse"])
        except KeyboardInterrupt:
            raise
        except:
            continue
        dev_data.append((jobj["text_tokens"], jobj["text_parse"], para["para_tokens"], para["para_parse"]))
    
logger.info(f"#Dev Instances: {len(dev_data)}")

fp1 = open(os.path.join(args.output, "train_src.txt"), "w")
fp2 = open(os.path.join(args.output, "train_src.parse"), "w")
fp3 = open(os.path.join(args.output, "train_tgt.txt"), "w")
fp4 = open(os.path.join(args.output, "train_tgt.parse"), "w")

for data in train_data:
    fp1.write(data[0]+"\n")
    fp2.write(data[1]+"\n")
    fp3.write(data[2]+"\n")
    fp4.write(data[3]+"\n")

fp1.close()
fp2.close()
fp3.close()
fp4.close()

fp1 = open(os.path.join(args.output, "dev_src.txt"), "w")
fp2 = open(os.path.join(args.output, "dev_src.parse"), "w")
fp3 = open(os.path.join(args.output, "dev_tgt.txt"), "w")
fp4 = open(os.path.join(args.output, "dev_tgt.parse"), "w")

for data in dev_data:
    fp1.write(data[0]+"\n")
    fp2.write(data[1]+"\n")
    fp3.write(data[2]+"\n")
    fp4.write(data[3]+"\n")

fp1.close()
fp2.close()
fp3.close()
fp4.close()

    
