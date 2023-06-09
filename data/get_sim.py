import os, logging, pprint, json
import torch
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from transformers import AutoModel, AutoTokenizer
from scipy.spatial.distance import cosine
import ipdb

# configuration
parser = ArgumentParser()
parser.add_argument('--data', default="./merge/paranmt_out4.txt")
args = parser.parse_args()

# logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)
logger.info(f"\n{pprint.pformat(vars(args), indent=4)}")


logger.info(f"loading {args.data}")
with open(args.data) as fp:
    lines = fp.readlines()
    
tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
model.cuda()

data = []
for line in tqdm(lines):
    jobj = json.loads(line)
    texts = [jobj["text"]] + jobj["paraphrases"]
    
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    inputs = {"input_ids": inputs["input_ids"].cuda(), "token_type_ids": inputs["token_type_ids"].cuda(), "attention_mask": inputs["attention_mask"].cuda()}

    # Get the embeddings
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    
    embeddings = embeddings.detach().cpu().numpy()
    sims = [1 - cosine(embeddings[0], embeddings[i]) for i in range(1, len(jobj["paraphrases"])+1)]
    dt = {
        "text": jobj["text"], 
        "paraphrases": [{"para_text": p, "similarity": s} for p, s in zip(jobj["paraphrases"], sims)]
    }
    data.append(dt)
    
with open(args.data+".sim", "w") as fp:
    for dt in data:
        fp.write(json.dumps(dt)+"\n")
