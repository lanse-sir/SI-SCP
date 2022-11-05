# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help

import os
import re
import subprocess
import threading
import tempfile
import sacrebleu
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm import tqdm

cherry = SmoothingFunction()

METEOR_JAR = '../eval_tools/meteor-1.5.jar'
METEOR_DATA = '../eval_tools/paraphrase-en.gz'
MULTI_BLEU_PERL = '../eval_tools/multi-bleu.perl'
STANFORD_CORENLP = '../eval_tools/stanford-corenlp-full-2018-10-05'

from nltk.tree import Tree
from zss import simple_distance, Node


def load_file_sent(file_name, if_ref=True):
    sentences = []
    with open(file_name, 'r', encoding='utf-8') as f_r:
        for line in f_r:
            if if_ref:
                sentences.append([line.strip().split()])
            else:
                sentences.append(line.strip().split())
    return sentences


def compute_self_bleu_per_input(multi_preds):
    pair_bleu4_list = []
    nums = len(multi_preds)

    for i in range(nums - 1):
        for j in range(i + 1, nums):
            # bleu4 = corpus_bleu([[multi_preds[j]]], [multi_preds[i]], smoothing_function=cherry.method3)
            bleu4 = sacrebleu.sentence_bleu(multi_preds[i], [multi_preds[j]])
            pair_bleu4_list.append(bleu4.score)
    return pair_bleu4_list


def compute_self_bleu(preds):
    pair_bleu4_list = []
    
    valid = 0
    for i in tqdm(range(len(preds))):
        if len(preds[i]) < 2:
            continue
        valid += 1
        pair_bleu4_list.extend(compute_self_bleu_per_input(preds[i]))

    return sum(pair_bleu4_list) / len(pair_bleu4_list)
    #return sum(pair_bleu4_list)/valid


def compute_bleu(refs, preds):
    bleu1 = corpus_bleu(refs, preds, weights=(1.0, 0, 0, 0), smoothing_function=cherry.method1)

    bleu2 = corpus_bleu(refs, preds, weights=(0.5, 0.5, 0, 0), smoothing_function=cherry.method1)
    bleu3 = corpus_bleu(refs, preds, weights=(0.33, 0.33, 0.34, 0), smoothing_function=cherry.method1)
    bleu4 = corpus_bleu(refs, preds, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=cherry.method1)
    return bleu1, bleu2, bleu3, bleu4


def enc(s):
    return s.encode('utf-8')


def dec(s):
    return s.decode('utf-8')


def run_multi_bleu(input_file, reference_file):
    bleu_output = subprocess.check_output(
        "perl {} -lc {} < {}".format(MULTI_BLEU_PERL, reference_file, input_file),
        stderr=subprocess.STDOUT, shell=True).decode('utf-8')
    bleu = float(
        bleu_output.strip().split("\n")[-1]
            .split(",")[0].split("=")[1][1:])
    return bleu


class Meteor:
    def __init__(self):
        self.meteor_cmd = ['java', '-jar', '-Xmx2G', METEOR_JAR,
                           '-', '-', '-stdio', '-l', 'en', '-norm', '-a',
                           METEOR_DATA]
        self.meteor_p = subprocess.Popen(
            self.meteor_cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        # Used to guarantee thread safety
        self.lock = threading.Lock()

    def compute_score(self, gts, res):
        assert (gts.keys() == res.keys())
        imgIds = gts.keys()
        scores = []

        eval_line = 'EVAL'
        self.lock.acquire()
        for i in imgIds:
            assert (len(res[i]) == 1)
            stat = self._stat(res[i][0], gts[i])
            eval_line += ' ||| {}'.format(stat)

        self.meteor_p.stdin.write(enc('{}\n'.format(eval_line)))
        self.meteor_p.stdin.flush()
        for i in range(0, len(imgIds)):
            scores.append(dec(float(self.meteor_p.stdout.readline().strip())))
        score = float(dec(self.meteor_p.stdout.readline().strip()))
        self.lock.release()

        return score, scores

    def _stat(self, hypothesis_str, reference_list):
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write(enc(score_line + "\n"))
        self.meteor_p.stdin.flush()
        return dec(self.meteor_p.stdout.readline()).strip()

    def _score(self, hypothesis_str, reference_list):
        # self.lock.acquire()
        with self.lock:
            # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
            hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
            score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
            self.meteor_p.stdin.write(enc(score_line + "\n"))
            self.meteor_p.stdin.flush()
            stats = dec(self.meteor_p.stdout.readline().strip())
            eval_line = 'EVAL ||| {}'.format(stats)
            # EVAL ||| stats
            self.meteor_p.stdin.write(enc('{}\n'.format(eval_line)))
            self.meteor_p.stdin.flush()
            score = float(dec(self.meteor_p.stdout.readline()).strip())
            # bug fix: there are two values returned by the jar file, one average, and one all, so do it twice
            # thanks for Andrej for pointing this out
            score = float(dec(self.meteor_p.stdout.readline().strip()))
        # self.lock.release()
        return score

    def __del__(self):
        self.lock.acquire()
        self.meteor_p.stdin.close()
        self.meteor_p.kill()
        self.meteor_p.wait()
        self.lock.release()


def deleaf(parse_string):
    tree = Tree.fromstring(parse_string.strip(), read_leaf=lambda s: "")
    for sub in tree.subtrees():
        for n, child in enumerate(sub):
            if isinstance(child, str):
                continue
            if len(list(child.subtrees(filter=lambda x: x.label() == '-NONE-'))) == len(child.leaves()):
                del sub[n]
    oneline = tree.pformat(margin=10000, parens=[" ( ", " ) "])
    oneline = re.sub(' +', ' ', oneline)
    return oneline


def extract_parses(fname):
    # extract parses from corenlp output
    # based on https://github.com/miyyer/scpn/blob/master/read_paranmt_parses.py
    with open(fname, 'r', encoding='utf-8') as f:

        count = 0
        sentences = []
        data = {'tokens': [], 'pos': [], 'parse': '', 'deps': []}
        for idx, line in enumerate(f):
            if idx <= 1:
                continue
            if line.startswith('Sentence #'):
                new_sent = True
                new_pos = False
                new_parse = False
                new_deps = False
                if idx == 2:
                    continue

                sentences.append(data)
                count += 1

                data = {'tokens': [], 'pos': [], 'parse': '', 'deps': []}

            # read original sentence
            elif new_sent:
                new_sent = False
                new_pos = True

            elif new_pos and line.startswith("Tokens"):
                continue

            # read POS tags
            elif new_pos and line.startswith('[Text='):
                line = line.strip().split()
                w = line[0].split('[Text=')[-1]
                pos = line[-1].split('PartOfSpeech=')[-1][:-1]
                data['tokens'].append(w)
                data['pos'].append(pos)

            # start reading const parses
            elif (new_pos or new_parse) and len(line.strip()):
                if line.startswith("Constituency parse"):
                    continue
                new_pos = False
                new_parse = True
                data['parse'] += ' ' + line.strip()

            # start reading deps
            elif (new_parse and line.strip() == "") or \
                    line.startswith("Dependency Parse"):
                new_parse = False
                new_deps = True

            elif new_deps and len(line.strip()):
                line = line.strip()[:-1].split('(', 1)
                rel = line[0]
                x1, x2 = line[1].split(', ')
                x1 = x1.replace("'", "")
                x2 = x2.replace("'", "")
                x1 = int(x1.rsplit('-', 1)[-1])
                x2 = int(x2.rsplit('-', 1)[-1])
                data['deps'].append((rel, x1 - 1, x2 - 1))

            else:
                new_deps = False

        sentences.append(data)

    return sentences


class stanford_parsetree_extractor:
    def __init__(self):
        self.stanford_corenlp_path = os.path.join(STANFORD_CORENLP, "*")
        print("standford corenlp path:", self.stanford_corenlp_path)
        self.output_dir = tempfile.TemporaryDirectory()
        self.cmd = ['java', '-cp', self.stanford_corenlp_path,
                    '-Xmx8G', 'edu.stanford.nlp.pipeline.StanfordCoreNLP',
                    '-annotators', 'tokenize,ssplit,pos,parse',
                    '-ssplit.eolonly', '-outputFormat', 'text',
                    '-outputDirectory', self.output_dir.name,
                    '-file', None]

    def run(self, file):
        print("parsing file:", file)
        self.cmd[-1] = file
        out = subprocess.run(
            self.cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        print(out)
        parsed_file = \
            os.path.join(
                self.output_dir.name,
                os.path.split(file)[1] + ".out")
        return [deleaf(e['parse']).strip() for e in extract_parses(parsed_file)]

    def cleanup(self):
        self.output_dir.cleanup()


def build_tree(s):
    old_t = Tree.fromstring(s)
    new_t = Node("S")

    def create_tree(curr_t, t):
        if t.label() and t.label() != "S":
            new_t = Node(t.label())
            curr_t.addkid(new_t)
        else:
            new_t = curr_t
        for i in t:
            if isinstance(i, Tree):
                create_tree(new_t, i)

    create_tree(new_t, old_t)
    return new_t


def strdist(a, b):
    if a == b:
        return 0
    else:
        return 1


def compute_tree_edit_distance(pred_parse, ref_parse):
    return simple_distance(
        build_tree(ref_parse), build_tree(pred_parse), label_dist=strdist)
