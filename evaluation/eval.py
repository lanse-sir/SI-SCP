import sys
import os
sys.path.append("/nfs/users/yangerguang/chuangxin/synpg_transformer/")
sys.path.append("/nfs/users/yangerguang/chuangxin/synpg_transformer/eval_tools/")
#sys.path.append("../../../")

#print(sys.path)
import argparse
import rouge

from autocg.utils_file import extract_template
from eval_utils import Meteor, stanford_parsetree_extractor, \
    compute_tree_edit_distance
from eval_utils import run_multi_bleu, compute_bleu, load_file_sent
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', '-i', type=str)
parser.add_argument('--ref_file', '-r', type=str)
parser.add_argument('--ted', action='store_true')
parser.add_argument("--ht", type=int, default=100)
args = parser.parse_args()

height = args.ht - 2

n_ref_line = len(list(open(args.ref_file)))
n_inp_line = len(list(open(args.input_file)))
print("#lines - ref: {}, inp: {}".format(n_ref_line, n_inp_line))
assert n_inp_line == n_ref_line, \
    "#ref {} != #inp {}".format(n_ref_line, n_inp_line)

preds = load_file_sent(args.input_file, if_ref=False)
refs = load_file_sent(args.ref_file, "sent")
bleu1, bleu2, bleu3, bleu4 = compute_bleu(refs, preds)

print("bleu-1: {:.4f}, bleu-2: {:.4f}, bleu-3: {:.4f}, bleu-4: {:.4f}".format(bleu1, bleu2, bleu3, bleu4))

bleu_score = run_multi_bleu(args.input_file, args.ref_file)
print("bleu", bleu_score)

if args.ted:
    spe = stanford_parsetree_extractor()
    input_parses = spe.run(os.path.join("/nfs/users/yangerguang/chuangxin/synpg_transformer/", args.input_file))
    ref_parses = spe.run(os.path.join("/nfs/users/yangerguang/chuangxin/synpg_transformer/", args.ref_file))
    spe.cleanup()
    assert len(input_parses) == n_inp_line
    assert len(ref_parses) == n_inp_line

    all_meteor = []
    all_ted = []
    all_rouge1 = []
    all_rouge2 = []
    all_rougel = []
    preds = []

    rouge_eval = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                             max_n=2,
                             limit_length=True,
                             length_limit=100,
                             length_limit_type='words',
                             apply_avg=False,
                             apply_best=False,
                             alpha=0.5,  # Default F1_score
                             weight_factor=1.2,
                             stemming=True)
    meteor = Meteor()
    pbar = tqdm(zip(open(args.input_file),
                    open(args.ref_file),
                    input_parses,
                    ref_parses))

    for input_line, ref_line, input_parse, ref_parse in pbar:
        input_parse = extract_template(input_parse, height)
        ref_parse = extract_template(ref_parse, height)
        ted = compute_tree_edit_distance(input_parse, ref_parse)
        ms = meteor._score(input_line.strip(), [ref_line.strip()])
        rs = rouge_eval.get_scores([input_line.strip()], [ref_line.strip()])

        all_rouge1.append(rs['rouge-1'][0]['f'][0])
        all_rouge2.append(rs['rouge-2'][0]['f'][0])
        all_rougel.append(rs['rouge-l'][0]['f'][0])
        all_meteor.append(ms)
        all_ted.append(ted)
        pbar.set_description(
            "bleu: {:.3f}, rouge-1: {:.3f}, rouge-2: {:.3f}, "
            "rouge-l: {:.3f}, meteor: {:.3f}, syntax-TED: {:.3f}".format(
                bleu_score,
                sum(all_rouge1) / len(all_rouge1) * 100,
                sum(all_rouge2) / len(all_rouge1) * 100,
                sum(all_rougel) / len(all_rouge1) * 100,
                sum(all_meteor) / len(all_meteor) * 100,
                sum(all_ted) / len(all_ted)))

    print(
        "bleu: {:.3f}, rouge-1: {:.3f}, rouge-2: {:.3f}, "
        "rouge-l: {:.3f}, meteor: {:.3f}, syntax-TED: {:.3f}".format(
            bleu_score,
            sum(all_rouge1) / len(all_rouge1) * 100,
            sum(all_rouge2) / len(all_rouge1) * 100,
            sum(all_rougel) / len(all_rouge1) * 100,
            sum(all_meteor) / len(all_meteor) * 100,
            sum(all_ted) / len(all_ted)))
else:
    all_meteor = []
    all_rouge1 = []
    all_rouge2 = []
    all_rougel = []
    preds = []

    rouge_eval = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                             max_n=2,
                             limit_length=True,
                             length_limit=100,
                             length_limit_type='words',
                             apply_avg=False,
                             apply_best=False,
                             alpha=0.5,  # Default F1_score
                             weight_factor=1.2,
                             stemming=True)
    meteor = Meteor()
    pbar = tqdm(zip(open(args.input_file),
                    open(args.ref_file)))

    for input_line, ref_line in pbar:
        # ted = compute_tree_edit_distance(input_parse, ref_parse)
        ms = meteor._score(input_line.strip(), [ref_line.strip()])
        rs = rouge_eval.get_scores([input_line.strip()], [ref_line.strip()])

        all_rouge1.append(rs['rouge-1'][0]['f'][0])
        all_rouge2.append(rs['rouge-2'][0]['f'][0])
        all_rougel.append(rs['rouge-l'][0]['f'][0])
        all_meteor.append(ms)
        # all_ted.append(ted)
        pbar.set_description(
            "bleu: {:.3f}, rouge-1: {:.3f}, rouge-2: {:.3f}, "
            "rouge-l: {:.3f}, meteor: {:.3f}".format(
                bleu_score,
                sum(all_rouge1) / len(all_rouge1) * 100,
                sum(all_rouge2) / len(all_rouge1) * 100,
                sum(all_rougel) / len(all_rouge1) * 100,
                sum(all_meteor) / len(all_meteor) * 100,
                # sum(all_ted) / len(all_ted)
            ))

        print(
            "bleu: {:.3f}, rouge-1: {:.3f}, rouge-2: {:.3f}, "
            "rouge-l: {:.3f}, meteor: {:.3f}".format(
                bleu_score,
                sum(all_rouge1) / len(all_rouge1) * 100,
                sum(all_rouge2) / len(all_rouge1) * 100,
                sum(all_rougel) / len(all_rouge1) * 100,
                sum(all_meteor) / len(all_meteor) * 100,
                # sum(all_ted) / len(all_ted)
            ))
