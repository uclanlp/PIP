import os, sys, json, logging, time, pprint, _jsonnet, subprocess, random
import numpy as np
import torch
import stanza
from argparse import ArgumentParser, Namespace
from zss import simple_distance, Node
from nltk.tree import Tree, ParentedTree
from tqdm import tqdm
import ipdb

# configuration
parser = ArgumentParser()
parser.add_argument('--gold', required=False)
parser.add_argument('--pred', required=False)
args = parser.parse_args()

def run_multi_bleu(input_file, reference_file):
    MULTI_BLEU_PERL = 'src/evaluation/apps/multi-bleu.perl'
    bleu_output = subprocess.check_output(
        "./{} -lc {} < {}".format(MULTI_BLEU_PERL, reference_file, input_file),
        stderr=subprocess.STDOUT, shell=True).decode('utf-8')
    bleu = float(bleu_output.strip().split("\n")[-1].split(",")[0].split("=")[1][1:])
    return bleu

def construct_tree(old_t, level):
    new_t = Node(old_t.label)
    if level > 0:
        childs = [construct_tree(child, level-1) for child in old_t.children if len(child.children) > 0]
        for child in childs:
            new_t.addkid(child)
    return new_t

def tree2str(tree):
    def tree2str_(tree, s):
        s.append("(")
        s.append(tree.label)
        for child in tree.children:
            tree2str_(child, s)
        s.append(")")
        return s
    return " ".join(tree2str_(tree, []))

def get_tmpl(tree, level):
    def get_tmpl_(tree, level, s):
        s.append("(")
        s.append(tree.label)
        if level > 0:
            for child in tree.children:
                get_tmpl_(child, level-1, s)
        s.append(")")
        return s
    return " ".join(get_tmpl_(tree, level, []))

def strdist(a, b):
    if a == b:
        return 0
    else:
        return 1

def compute_tree_edit_distance(pred_parse, ref_parse):
    return simple_distance(build_tree(ref_parse), build_tree(pred_parse), label_dist=strdist)

def compute_tree_edit_distance3(pred_parse, ref_parse):
    return simple_distance(build_tree3(ref_parse), build_tree3(pred_parse), label_dist=strdist)


with open(args.gold) as fp:
    golds = fp.readlines()

with open(args.pred) as fp:
    preds = fp.readlines()
    
assert len(golds) == len(preds)

# golds = golds[:30]
# preds = preds[:30]

parser = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')


exact = []
tedf = []
ted2 = []
ted3 = []

i = 0
for gold, pred in tqdm(zip(golds, preds), total=len(golds)):
    gold = gold.strip()
    pred = pred.strip()
    
    g_obj = parser(gold)
    p_obj = parser(pred)
    
    g_tree = construct_tree(g_obj.sentences[0].constituency, 10000000)
    p_tree = construct_tree(p_obj.sentences[0].constituency, 10000000)
    
    g_tree3 = construct_tree(g_obj.sentences[0].constituency, 3)
    p_tree3 = construct_tree(p_obj.sentences[0].constituency, 3)
    
    g_tree2 = construct_tree(g_obj.sentences[0].constituency, 2)
    p_tree2 = construct_tree(p_obj.sentences[0].constituency, 2)
    
    g_tmpl = tree2str(g_tree2)
    p_tmpl = tree2str(p_tree2)
    
    
    s1 = g_tmpl==p_tmpl
    s2 = simple_distance(g_tree, p_tree, label_dist=strdist)
    s3 = simple_distance(g_tree2, p_tree2, label_dist=strdist)
    s4 = simple_distance(g_tree3, p_tree3, label_dist=strdist)
    print(s1)
    print(s2)
    print(s3)
    print(s4)
    
    # if i==25:
    #     ipdb.set_trace()
    
    i += 1
    
    exact.append(g_tmpl==p_tmpl)
    tedf.append(simple_distance(g_tree, p_tree, label_dist=strdist))
    ted2.append(simple_distance(g_tree2, p_tree2, label_dist=strdist))
    ted3.append(simple_distance(g_tree3, p_tree3, label_dist=strdist))
    
print(f"BLEU: {run_multi_bleu(args.pred, args.gold)}")
print(f"TMA: {100.0*np.mean(exact)}")
print(f"TED-F: {np.mean(tedf)}")
print(f"TED-2: {np.mean(ted2)}")
print(f"TED-3: {np.mean(ted3)}")
