import os, logging, pprint, json
import stanza
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
import ipdb

# configuration
parser = ArgumentParser()
parser.add_argument('--data', default="./merge/parabank2_out3.txt")
args = parser.parse_args()

# logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)
logger.info(f"\n{pprint.pformat(vars(args), indent=4)}")

output_fn = f"{args.data}.parse"

parser = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')

logger.info(f"Loading {args.data}")
with open(args.data) as fp:
    lines = fp.readlines()
    
data = []
n_fail = 0
for line in tqdm(lines, ascii=True):
    jobj = json.loads(line)
    
    text = jobj["text"]
    
    try:
        text_obj = parser(text)
    except KeyboardInterrupt:
        raise
    except:
        n_fail += 1
        dt = {
            "text": text, 
        }
        data.append(dt)
        continue
    
    paraphrases = []
    for para in jobj["paraphrases"]:
        try:
            para_obj = parser(para)
            paraphrases.append({
                "para_text": para, 
                "para_tokens": " ".join([t.text for t in para_obj.sentences[0].tokens]), 
                "para_parse": str(para_obj.sentences[0].constituency), 
            })
        except KeyboardInterrupt:
            raise
        except:
            continue
    
    if len(paraphrases) == 0:
        n_fail += 1
        dt = {
            "text": text, 
        }
        data.append(dt)
        continue
        
    dt = {
        "text": text, 
        "text_tokens": " ".join([t.text for t in text_obj.sentences[0].tokens]), 
        "text_parse": str(text_obj.sentences[0].constituency), 
        "paraphrases": paraphrases
    }
    data.append(dt)
    
logger.info(f"Number of failed examples: {n_fail}")

logger.info(f"Saving {output_fn}")
with open(output_fn, "w") as fp:
    for dt in data:
        fp.write(json.dumps(dt)+"\n")

