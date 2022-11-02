import sys
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")

from autocg.load_file import load_sentence, list_to_file
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pred_file", type=str)
parser.add_argument("--src_file", type=str)
args = parser.parse_args()


def main(args):
    preds = load_sentence(args.pred_file)
    srcs = load_sentence(args.src_file)
    
    assert len(preds) == len(srcs)
    for i in range(len(preds)):
        if len(preds[i])==0:
            preds[i] = srcs[i]
    
    list_to_file(args.pred_file, preds)


if __name__ == '__main__':
    main(args)