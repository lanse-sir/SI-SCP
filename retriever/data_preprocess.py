import os
import pickle
# import numpy as np
import sys
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

from autocg.load_file import load_file_sent_or_parser
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str)
# parser.add_argument("--output", type=str, default="data/paranmt-350k-4~30/train_parse-5.pkl")
parser.add_argument("--level", type=str, default=4)
parser.add_argument("--file", type=str, default="train")
parser.add_argument("--remove_pos", action="store_true")
args = parser.parse_args()

if args.remove_pos:
    tag = "_remove-pos"
else:
    tag = ""

data_root = os.path.join(args.root, args.file)
src_sent_file = os.path.join(data_root, args.file + "_src.txt")
src_pos_file = os.path.join(data_root, args.file + f"_src_parse{tag}-{args.level}.txt")

tgt_parse_file = os.path.join(data_root, args.file + f"_tgt_parse{tag}-{args.level}.txt")

src_sents = load_file_sent_or_parser(src_sent_file, "sent")
src_pos = load_file_sent_or_parser(src_pos_file, "sent")
tgt_parses = load_file_sent_or_parser(tgt_parse_file, "parse")

original_nums = len(src_sents)
print("Original Nums: ", original_nums)
assert len(src_sents) == len(src_pos) == len(tgt_parses)
# for idx, (src, pos) in enumerate(zip(src_sents, src_pos)):
#     if len(src) != len(pos):
#         print(idx+1)
#         del src_sents[idx]
#         del src_pos[idx]
#         del tgt_parses[idx]

now_nums = len(src_sents)
print("Remove {} sentences. ".format(original_nums - now_nums))

output = f"{args.root}/{args.file}_parse{tag}-{args.level}.pkl"
f = open(output, "wb")
h5 = dict()
h5["src_sents"] = src_sents
h5["src_pos"] = src_pos
h5["tgt_parse"] = tgt_parses

pickle.dump(h5, f)
print("Save file to {}".format(output))
