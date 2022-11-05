import sys
sys.path.append('../../')
import argparse
import rouge
from autocg.load_file import load_sentence


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


def candidates_load(path):
    candidates = []
    with open(path, 'r', encoding='utf-8') as f_r:
        for line in f_r:
            candidates.append(line.strip().split('\t'))
    return candidates


def select_rouge(src, candidates):
    max_rouge = -1
    max_idx = None
    for i, cand in enumerate(candidates):
        # rouge_score = rouge_eval.get_scores([cand], [src])
        rouge_score = rouge_eval.get_scores([cand], [src])['rouge-1'][0]['f'][0]
        if rouge_score > max_rouge:
            max_rouge = rouge_score
            max_idx = i
    return max_idx


def main(args):
    src_sents = load_sentence(args.src)
    candidates = candidates_load(args.gene)

    final_sents = []
    for src, cands in zip(src_sents, candidates):
        max_idx = select_rouge(src, cands)
        final_sents.append(cands[max_idx])

    with open(args.save_file, 'w', encoding='utf-8') as f_w:
        for sent in final_sents:
            f_w.write(sent + '\n')
    
    print('Done ......')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gene', type=str)
    parser.add_argument('--src', type=str)
    parser.add_argument('--save_file', type=str)
    args = parser.parse_args()
    main(args)














