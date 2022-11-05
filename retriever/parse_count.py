import argparse
import sys

sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

from collections import Counter
from autocg.load_file import load_file_sent_or_parser, save_to_file


parser = argparse.ArgumentParser()
parser.add_argument("--inp_file", type=str)
parser.add_argument("--topk", type=int, default=10)
parser.add_argument("--s", type=str)
args = parser.parse_args()

parses = load_file_sent_or_parser(args.inp_file, type="parse")

parse_counts = Counter(parses)

top_parses = []
for p, n in parse_counts.most_common(args.topk):
    top_parses.append(p)
    print(p, "----", n)

if args.s is not None:
    print(f"Save to {args.s}")
    save_to_file(top_parses, args.s)