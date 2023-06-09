import os, logging, pprint, json
import stanza
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
import ipdb

# configuration
parser = ArgumentParser()
parser.add_argument('--data_dir', required=True)
args = parser.parse_args()

# logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)
logger.info(f"\n{pprint.pformat(vars(args), indent=4)}")

files = [("train/src.pseudo", "train/src.pseudo.parse")]
# files = [("train/src-small.pseudo", "train/src-small.pseudo.parse")]
# files = [("train/src-medium.pseudo", "train/src-medium.pseudo.parse")]

parser = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')

for text_fn, parse_fn in files:
    try:
        logger.info(f"Processing {os.path.join(args.data_dir, text_fn)}")
        with open(os.path.join(args.data_dir, text_fn)) as fp:
            lines = fp.readlines()
    except:
        logger.info(f"{os.path.join(args.data_dir, text_fn)} does not exist.")
        continue
    
    data = []
    for line in tqdm(lines, ascii=True):
        raw = json.loads(line)
        
        pseudos = []
        for text_p in raw["pseudo"]:
            try:
                obj = parser(text_p)
                text = " ".join([t.text for t in obj.sentences[0].tokens])
                parse = str(obj.sentences[0].constituency)
                pseudos.append((text, parse))
            except KeyboardInterrupt:
                raise
            except:
                continue
            
        dt = {"text": raw["text"], "pseudo": pseudos}
        data.append(dt)
            
    logger.info(f"Saving {os.path.join(args.data_dir, parse_fn)}")
    with open(os.path.join(args.data_dir, parse_fn), "w") as fp:
        for dt in data:
            fp.write(json.dumps(dt)+"\n")

        
