import os, logging, pprint, json
import amrlib, penman
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
import ipdb

# configuration
parser = ArgumentParser()
parser.add_argument('--data_dir', required=True)
parser.add_argument('--amr_parser_dir', default="./amr_models/model_parse_spring-v0_1_0/")
parser.add_argument('--amr_decoder_dir', default="./amr_models/model_generate_t5wtense-v0_1_0/")
args = parser.parse_args()

# logger
logging.basicConfig(level=logging.WARN, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)
logger.warning(f"\n{pprint.pformat(vars(args), indent=4)}")

files = [("train/src.txt", "train/src.pseudo")]

amr_parser = amrlib.load_stog_model(args.amr_parser_dir)
amr_decoder = amrlib.load_gtos_model(args.amr_decoder_dir)

for text_fn, pseudo_fn in files:
    try:
        logger.warning(f"Processing {os.path.join(args.data_dir, text_fn)}")
        with open(os.path.join(args.data_dir, text_fn)) as fp:
            lines = fp.readlines()
    except:
        logger.warning(f"{os.path.join(args.data_dir, text_fn)} does not exist.")
        continue
    
    data = []
    for line in tqdm(lines, ascii=True):
        text = line.strip()
        
        try:
            graphs = amr_parser.parse_sents([text])
            graph = graphs[0]
            pgraph = penman.decode(graph)
            candidate_tops = pgraph.variables()
            candidate_tops.remove(pgraph.top)
            new_graphs = [penman.encode(pgraph, top=t) for t in candidate_tops]
            gen_sents, _ = amr_decoder.generate(new_graphs, disable_progress=True, use_tense=False)
            gen_sents = sorted(set([s.lower() for s in gen_sents]))
        except:
            gen_sents = []
        
        dt = {"text": text, "pseudo": gen_sents}
        data.append(dt)
    
    logger.info(f"Saving {os.path.join(args.data_dir, pseudo_fn)}")
    with open(os.path.join(args.data_dir, pseudo_fn), "w") as fp:
        for dt in data:
            fp.write(json.dumps(dt)+"\n")

