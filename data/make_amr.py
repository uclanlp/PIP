import os, logging, pprint, json
import amrlib, penman
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
import ipdb

# configuration
parser = ArgumentParser()
parser.add_argument('--data', default="./merge/paranmt_out3.txt")
parser.add_argument('--start', type=int, required=True)
parser.add_argument('--end', type=int, required=True)
parser.add_argument('--amr_parser_dir', default="../amr_models/model_parse_spring-v0_1_0/")
parser.add_argument('--amr_decoder_dir', default="../amr_models/model_generate_t5wtense-v0_1_0/")
parser.add_argument('--output', default="./merge/paraamr_out3.txt")
args = parser.parse_args()

# logger
logging.basicConfig(level=logging.WARN, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)
logger.warning(f"\n{pprint.pformat(vars(args), indent=4)}")

amr_parser = amrlib.load_stog_model(args.amr_parser_dir)
amr_decoder = amrlib.load_gtos_model(args.amr_decoder_dir)

output_fn = f"{args.output}.{args.start}.{args.end}"

with open(output_fn, "w") as fp:
    pass

with open(args.data) as fp:
    lines = fp.readlines()
lines = lines[args.start:args.end]
    
fp = open(output_fn, "a")
n_data = 0
for line in tqdm(lines, ascii=True):
    jobj = json.loads(line)
    text = jobj["text"]
    
    try:
        graphs = amr_parser.parse_sents([text])
        graph = graphs[0]
        pgraph = penman.decode(graph)
        candidate_tops = pgraph.variables()
        candidate_tops.remove(pgraph.top)
        new_graphs = [penman.encode(pgraph, top=t) for t in candidate_tops]
        gen_sents, _ = amr_decoder.generate(new_graphs, disable_progress=True, use_tense=False)
        gen_sents = sorted(set([s.lower() for s in gen_sents]))
    except KeyboardInterrupt:
        raise
    except:
        gen_sents = []
    
    dt = {"text": text, "paraphrases": gen_sents}
    fp.write(json.dumps(dt)+"\n")
    n_data += 1

logger.warning(f"load {n_data}/{len(lines)} examples")
