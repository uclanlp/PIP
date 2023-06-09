import os, logging, pprint, json
import numpy as np
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer
from argparse import ArgumentParser, Namespace
import ipdb

# configuration
parser = ArgumentParser()
parser.add_argument('--input', default="./merge/paraamr_out5.txt.parse")
parser.add_argument('--output', default="./merge/paraamr_out5.txt.parse.permy")
parser.add_argument('--start', type=int, default=0)
args = parser.parse_args()

# logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)
logger.info(f"\n{pprint.pformat(vars(args), indent=4)}")


logger.info(f"loading {args.input}")
with open(args.input) as fp:
    lines = fp.readlines()

    
model = AutoModelForCausalLM.from_pretrained("gpt2-large").cuda()
tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
loss_fct = CrossEntropyLoss()

fp = open(args.output, "a+")
for line in tqdm(lines[args.start:], ncols=100, ascii=True):
    jobj = json.loads(line)
    
    paraphrases = [{"para_text": para} for para in jobj["paraphrases"] if len(para.split(" ")) > 2]
    if len(paraphrases) > 0:
        input_texts = [para["para_text"] for para in paraphrases]
        for i in range(len(paraphrases)):
            inputs = tokenizer([input_texts[i]], return_tensors="pt")
            input_ids = inputs['input_ids'].cuda()
            attn_mask = inputs['attention_mask'].cuda()
            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            
            shift_logits = outputs.logits[0, :-1, :].contiguous()
            shift_labels = input_ids[0, 1:].contiguous()
            
            paraphrases[i]["perplexity"] = np.exp(loss_fct(shift_logits, shift_labels).item())
    
    jobj["paraphrases"] = paraphrases
    fp.write(json.dumps(jobj)+"\n")

fp.close()
