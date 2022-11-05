import argparse
import sacrebleu
from eval_utils import compute_self_bleu
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction


# cherry = SmoothingFunction()
# s1 = "you are beautiful ."
# s2 = "you are very beautiful ."
#
# result = sacrebleu.sentence_bleu(s2, [s1])
#
# print(result.score)
# print("Sacrebleu: ", sacrebleu.corpus_bleu([s2], [[s1]]))
# print("Sentencebleu: ", sacrebleu.sentence_bleu(s2, [s1]))
#
# print(sentence_bleu([s1.split()], s2.split(), smoothing_function=cherry.method3))


def detect_null_line(sent_list):
    flag = False
    for sent in sent_list:
        if sent == "":
            flag = True
            break
    return flag


def load_multi_column(fname):
    sents = []
    with open(fname, 'r', encoding='utf-8') as fr:
        for line in fr:
            sent_list = line.strip().split("\t")
            if not detect_null_line(sent_list):
                sents.append(sent_list)
    return sents

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp_file", type=str, default=None)
    args = parser.parse_args()

    # pred_file = "../baseline_work/sow-reap_generations.txt"

    preds = load_multi_column(args.inp_file)

    print("Non-Null Line numbers: ", len(preds))
    score = compute_self_bleu(preds)

    print(score)
