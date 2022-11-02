import sys

sys.path.append("../")
sys.path.append("../../")

import torch
from synpg_transformer import synpg_transformer
from train import auto_evaluate, load_vocab
from autocg.load_file import load_file_sent_or_parser
from generator import Translator
from data_init import trees_to_dict
import argparse

parser = argparse.ArgumentParser(
    description="Training Syntactic Text Generation Models",
    usage="generator.py [<args>] [-h | --help]")
parser.add_argument('--dev_parse', type=str, default=None)
parser.add_argument('--dev_src', type=str, default=None)
parser.add_argument('--dev_ref', type=str, default=None)
parser.add_argument('--sent', type=str)

parser.add_argument('--model_file', type=str)
parser.add_argument('--pos_vocab', type=str)
parser.add_argument('--eval_bs', type=int, default=32)
parser.add_argument('--ht', type=int, default=4)
parser.add_argument('--greedy', action="store_true")
parser.add_argument('--return_attn', action="store_true")

parser.add_argument('--beam_size', type=int, default=2)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--bpe', type=bool, default=True)
parser.add_argument('--remove_pos', action='store_true')
parser.add_argument('--sent_max_time_step', type=int, default=100)
opt = parser.parse_args()

opt.parse_pe=False
sents = load_file_sent_or_parser(opt.dev_src, "sent")
parses = load_file_sent_or_parser(opt.dev_parse, "parse")

# parses = trees_to_dict(parses)

pos_vocab = load_vocab(opt.pos_vocab)

inputs = list(zip(sents, parses))
print(inputs[0])
torch.cuda.set_device(opt.gpu)
params = torch.load(opt.model_file, map_location=lambda storage, loc: storage)
autocg_model = synpg_transformer(params['args'], params['word_vocab'], params['parse_vocab'])
autocg_model.load_state_dict(params['state_dict'])
if torch.cuda.is_available():
    autocg_model = autocg_model.cuda()

translator = Translator(model=autocg_model, beam_size=opt.beam_size, max_seq_len=100)

ori_bleu, ref_bleu = auto_evaluate(model=translator, test_data=inputs, word_vocab=params['word_vocab'],
                                   parse_vocab=params['parse_vocab'],
                                   pos_vocab=pos_vocab, opt=opt)
print('Ori BLEU=%.3f, Ref BLEU=%.3f' % (ori_bleu, ref_bleu))
