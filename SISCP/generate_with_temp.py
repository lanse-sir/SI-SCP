import sys
import copy
sys.path.append("../")
sys.path.append("../../")

import torch
from synpg_transformer import synpg_transformer
from train import diverse_generate, load_vocab
from autocg.load_file import load_file_sent_or_parser
from generator import Translator
from data_init import trees_to_dict
import argparse


def input_pair(sents, templates):
    datas = []
    n = len(templates) // len(sents)
    for idx, s in enumerate(sents):
        for temp in templates[idx*n:(idx+1)*n]:
            datas.append((s, temp))
    return datas


def input_pair_common(sents, templates):
    datas = []
    for s in sents:
        for temp in templates:
            t1 = copy.deepcopy(temp)
            datas.append((s, t1))
    return datas


parser = argparse.ArgumentParser(
    description="Training Syntactic Text Generation Models",
    usage="generator.py [<args>] [-h | --help]")
parser.add_argument('--dev_parse', type=str, default=None)
parser.add_argument('--dev_src', type=str, default=None)
parser.add_argument('--dev_ref', type=str, default=None)
parser.add_argument('--sent', type=str)

parser.add_argument('--model_file', type=str)
parser.add_argument('--pos_vocab', type=str, default=None)
parser.add_argument('--eval_bs', type=int, default=64)
parser.add_argument('--ht', type=int, default=4)
parser.add_argument('--greedy', type=bool, default=False)
parser.add_argument('--beam_size', type=int, default=2)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--bpe', type=bool, default=True)
parser.add_argument('--remove_pos', action='store_true')
parser.add_argument('--return_attn', action='store_true')
parser.add_argument('--sent_max_time_step', type=int, default=100)
opt = parser.parse_args()

sents = load_file_sent_or_parser(opt.dev_src, "sent")
parses = load_file_sent_or_parser(opt.dev_parse, "parse")

# string to dict.
parses = trees_to_dict(parses)
print("Sentence Nums: ", len(sents))
print("Template Nums: ", len(parses))


pos_vocab = None

if len(parses) > len(sents):
    print("Using Retrieved Template ......")
    inputs = input_pair(sents, parses)
else:
    print("Using Common Template ......")
    inputs = input_pair_common(sents, parses)

print("Input Size: ", len(inputs))
print(inputs[0])

torch.cuda.set_device(opt.gpu)
params = torch.load(opt.model_file, map_location=lambda storage, loc: storage)
autocg_model = synpg_transformer(params['args'], params['word_vocab'], params['parse_vocab'])
autocg_model.load_state_dict(params['state_dict'])
if torch.cuda.is_available():
    autocg_model = autocg_model.cuda()

translator = Translator(model=autocg_model, beam_size=opt.beam_size, max_seq_len=100)

print("Generating......")
diverse_generate(model=translator, test_data=inputs, word_vocab=params['word_vocab'],
                   parse_vocab=params['parse_vocab'],
                   pos_vocab=pos_vocab, opt=opt)
