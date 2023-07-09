import os, logging
from tqdm import tqdm
from nltk import ParentedTree, Tree
import ipdb
logger = logging.getLogger(__name__)

def load_special_tokens(file):
    with open(file) as fp:
        special_tokens = list(map(lambda x: f"<{x.strip()}>", fp.readlines()))
    return special_tokens

def load_data(src_sent_file, src_synt_file, tgt_sent_file, tgt_synt_file, tokenizer, config):
    logger.info(f"loading instances from {src_sent_file}, {src_synt_file}, {tgt_sent_file}, {tgt_synt_file}")
    with open(src_sent_file) as fp:
        src_sents = list(map(lambda x: x.strip(), fp.readlines()))[:300]
    with open(src_synt_file) as fp:
        src_synts = list(map(lambda x: x.strip(), fp.readlines()))[:300]
    with open(tgt_sent_file) as fp:
        tgt_sents = list(map(lambda x: x.strip(), fp.readlines()))[:300]
    with open(tgt_synt_file) as fp:
        tgt_synts = list(map(lambda x: x.strip(), fp.readlines()))[:300]
    20
    assert len(src_sents) == len(src_synts) == len(tgt_synts) == len(tgt_sents)
    data = []
    for src_sent, src_synt, tgt_sent, tgt_synt in tqdm(zip(src_sents, src_synts, tgt_sents, tgt_synts), total=len(src_sents), ncols=100, ascii=True):
        
        src_sent_len = len(tokenizer(src_sent)['input_ids'])
        if src_sent_len > config.max_src_sent_len:
            continue
        
        src_synt_len = len(tokenizer(src_synt)['input_ids'])
        if src_synt_len > config.max_src_synt_len:
            continue
            
        tgt_sent_len = len(tokenizer(tgt_sent)['input_ids'])
        if tgt_sent_len > config.max_tgt_sent_len:
            continue
        
        # ipdb.set_trace()
        tgt_synt = remove_leaves(tgt_synt)
        # tgt_synt = remove_leaves(trim_str(tgt_synt, config.trim_h))
        tgt_synt_len = len(tokenizer(tgt_synt)['input_ids'])
        # if tgt_synt_len > 100:
        #     print('tgt', tgt_synt_len)
        #     continue
        if tgt_synt_len > config.max_tgt_synt_len:
            continue
            
        data.append({"src_sent": src_sent, "src_synt": src_synt, "tgt_sent": tgt_sent, "tgt_synt": tgt_synt})
        
    logger.info(f"load {len(data)}/{len(src_sents)} instances")
    
    return data

def load_translation_data(src_sent_file, src_synt_file, tgt_sent_file,  tokenizer, config):
    logger.info(f"loading instances from {src_sent_file}, {src_synt_file}, {tgt_sent_file}")
    with open(src_sent_file) as fp:
        src_sents = list(map(lambda x: x.strip(), fp.readlines()))[:30000]
    with open(src_synt_file) as fp:
        src_synts = list(map(lambda x: x.strip(), fp.readlines()))[:30000]
    with open(tgt_sent_file) as fp:
        tgt_sents = list(map(lambda x: x.strip(), fp.readlines()))[:30000]
    20
    assert len(src_sents) == len(src_synts) == len(tgt_sents)
    data = []
    for src_sent, src_synt, tgt_sent in tqdm(zip(src_sents, src_synts, tgt_sents), total=len(src_sents), ncols=100, ascii=True):
        
        src_sent_len = len(tokenizer(src_sent)['input_ids'])
        if src_sent_len > config.max_src_sent_len:
            continue
        
        src_synt_len = len(tokenizer(src_synt)['input_ids'])
        if src_synt_len > config.max_src_synt_len:
            continue
            
        tgt_sent_len = len(tokenizer(tgt_sent)['input_ids'])
        if tgt_sent_len > config.max_tgt_sent_len:
            continue
            
        data.append({"src_sent": src_sent, "src_synt": src_synt, "tgt_sent": tgt_sent})
        
    logger.info(f"load {len(data)}/{len(src_sents)} instances")
    
    return data

def remove_leaves(s):
    def _remove_leaves(tree):
        for child in tree:
            if isinstance(child, ParentedTree):
                _remove_leaves(child)
            else:
                tree.remove(child)
    tree = ParentedTree.fromstring(s)
    _remove_leaves(tree)
    t = tree._pformat_flat(nodesep="", parens="()", quotes=False)
    t = t.replace(" )", ")")
    return t

    
def string_comma(string):
    start = 0
    new_string = ''
    while start < len(string):
        if string[start:].find(",") == -1:
            new_string += string[start:]
            break
        else:
            index = string[start:].find(",")
            if string[start - 2] != "(":
                new_string += string[start:start + index]
                new_string += " "
            else:
                new_string = new_string[:start - 1] + ", "
            start = start + index + 1
    return new_string

def clean_tuple_str(tuple_str):
    new_str_ls = []
    if len(tuple_str) == 1:
        new_str_ls.append(tuple_str[0])
    else:
        for i in str(tuple_str).split(", "):
            if i.count("'") == 2:
                new_str_ls.append(i.replace("'", ""))
            elif i.count("'") == 1:
                new_str_ls.append(i.replace("\"", ""))
    str_join = ' '.join(ele for ele in new_str_ls)
    return string_comma(str_join)

def to_tuple(lst):
    return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)

def trim_tree_nltk(root, height):
    try:
        root.label()
    except AttributeError:
        return

    if height < 1:
        return
    all_child_state = []
    #     print(root.label())
    all_child_state.append(root.label())

    if len(root) >= 1:
        for child_index in range(len(root)):
            child = root[child_index]
            if trim_tree_nltk(child, height - 1):
                all_child_state.append(trim_tree_nltk(child, height - 1))
    #                 print(all_child_state)
    return all_child_state

def trim_str(string, height):
    return clean_tuple_str(to_tuple(trim_tree_nltk(Tree.fromstring(string), height)))