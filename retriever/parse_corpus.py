import argparse
import sys

sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

from collections import Counter
from autocg.load_file import load_sentence, save_to_file

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str)
parser.add_argument('--level', type=int, default=4)
parser.add_argument('--threshold', type=int, default=15)
parser.add_argument('--remove_pos', action="store_true")
args = parser.parse_args()

if args.remove_pos:
    type = "_remove_pos"
else:
    type = ""

tgt_file = f"{args.root}/train/train_tgt_parse{type}-{args.level}.txt"

dev_tgt_file = f"{args.root}/dev/dev_tgt_parse{type}-{args.level}.txt"
test_tgt_file = f"{args.root}/test/test_tgt_parse{type}-{args.level}.txt"

# src_parse_tree = load_sentence(src_file)
tgt_parse_tree = load_sentence(tgt_file)

dev_tgt_tree = load_sentence(dev_tgt_file)
test_tgt_tree = load_sentence(test_tgt_file)

total = tgt_parse_tree + dev_tgt_tree + test_tgt_tree
print("Total Nums: ", len(total))

print("Dev: ", len(set(dev_tgt_tree) - set(total)))
print("Test: ", len(set(test_tgt_tree) - set(total)))

parse = Counter(total)

print("Total parse set: {}".format(len(parse)))
# for p, freq, in parse.most_common(10):
#     print(p, "  ", freq)

save_list = list()
for k, v in parse.items():
    if v >= args.threshold:
        save_list.append(k)

print("Save nums: {}, Remove nums: {}".format(len(save_list), len(parse) - len(save_list)))
save_file = f"{args.root}/parse{type}-{args.level}_set-{args.threshold}.txt"

save_to_file(save_list, save_file)
print("Save to {}".format(save_file))
