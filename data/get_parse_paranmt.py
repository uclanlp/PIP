import os, logging, pprint, json
import stanza
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
import ipdb

# configuration
parser = ArgumentParser()
parser.add_argument('--data', default="./merge/paranmt_out3.txt")
parser.add_argument('--output', default="./merge/paranmt_out3.txt.parse")
parser.add_argument('--start', type=int, required=True)
parser.add_argument('--end', type=int, required=True)
args = parser.parse_args()

# logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)
logger.info(f"\n{pprint.pformat(vars(args), indent=4)}")

output_fn = f"{args.output}.{args.start}.{args.end}"

parser = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')

logger.info(f"Loading {args.data}")
with open(args.data) as fp:
    lines = fp.readlines()
lines = lines[args.start:args.end]
    
data = []
n_fail = 0
for line in tqdm(lines, ascii=True):
    line = line.strip().split("\t")
    
    ref = line[0]
    par = line[1]
    
    try:
        ref_obj = parser(ref)
        par_obj = parser(par)
        dt = {
            "text": ref, 
            "text_tokens": " ".join([t.text for t in ref_obj.sentences[0].tokens]), 
            "text_parse": str(ref_obj.sentences[0].constituency), 
            "paraphrase": [{
                "paraphrase": par, 
                "paraphrase_tokens": " ".join([t.text for t in par_obj.sentences[0].tokens]), 
                "paraphrase_parse": str(par_obj.sentences[0].constituency), 
            }]
        }
    except KeyboardInterrupt:
        raise
    except:
        n_fail += 1
        dt = {
            "text": ref, 
            "paraphrase": [{
                "paraphrase": par, 
            }]
        }
        

    data.append(dt)
    
logger.info(f"Number of failed examples: {n_fail}")

logger.info(f"Saving {output_fn}")
with open(output_fn, "w") as fp:
    for dt in data:
        fp.write(json.dumps(dt)+"\n")

