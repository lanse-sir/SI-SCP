import argparse
import sys

sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

from autocg.load_file import load_file_sent_or_parser

parser = argparse.ArgumentParser()
parser.add_argument("--inp_file", type=str)
parser.add_argument("--target", type=str)
parser.add_argument("--topk",type=int)
args = parser.parse_args()

src_parses = load_file_sent_or_parser(args.inp_file, type="parse")
tgt_parses = load_file_sent_or_parser(args.target, type="parse")

tgt_parses = tgt_parses[:args.topk]
print("All parses: ", len(src_parses))
print("Common parses: ", len(tgt_parses))

correct = 0
for p in src_parses:
    if p in tgt_parses:
        correct += 1

print("Correct Nums:", correct)
print("Recall: ", correct / len(src_parses))
