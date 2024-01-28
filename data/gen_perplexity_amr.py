import os, logging, pprint, json
import evaluate_seq2seq
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
import ipdb

# configuration
parser = ArgumentParser()
parser.add_argument('--input', default="./merge/paraamr_out5.txt.parse")
parser.add_argument('--output', default="./merge/paraamr_out5.txt.parse.per")
parser.add_argument('--start', type=int, default=0)
args = parser.parse_args()

# logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)
logger.info(f"\n{pprint.pformat(vars(args), indent=4)}")


logger.info(f"loading {args.input}")
with open(args.input) as fp:
    lines = fp.readlines()

    
perplexity = evaluate_seq2seq.load("perplexity", module_type="metric")

fp = open(args.output, "a+")
for line in tqdm(lines[args.start:], ncols=100, ascii=True):
    jobj = json.loads(line)
    
    paraphrases = [{"para_text": para} for para in jobj["paraphrases"] if len(para.split(" ")) > 2]
    if len(paraphrases) > 0:
        input_texts = [para["para_text"] for para in paraphrases]
        results = perplexity.compute(model_id='gpt2', add_start_token=False, input_texts=input_texts)
        for i in range(len(paraphrases)):
            paraphrases[i]["perplexity"] = results["perplexities"][i]
    
    jobj["paraphrases"] = paraphrases
    fp.write(json.dumps(jobj)+"\n")

fp.close()
