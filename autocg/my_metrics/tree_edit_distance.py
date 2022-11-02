from nltk.tree import Tree
from zss import simple_distance, Node
import distance


def edit_distance(seq1, seq2):
    return distance.levenshtein(seq1, seq2)


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
        ref_parse, pred_parse, label_dist=strdist)


def corpus_tree_edit_distance(preds, refs):
    assert len(preds) == len(refs), 'Length is not same .'
    total_ted = []
    for i in range(len(refs)):
        # list or str object .
        if type(preds[i]) is list and type(refs[i]) is list:
            pred = ' '.join(preds[i])
            ref = ' '.join(refs[i])
        elif type(preds[i]) is str and type(refs[i]) is str:
            pred = preds[i]
            ref = refs[i]
        ted = compute_tree_edit_distance(pred_parse=build_tree(pred), ref_parse=build_tree(ref))
        if len(refs) <= 20:
            print("Sentence index [%d], Tree edit distance is [%d]" % (i + 1, ted))
        total_ted.append(ted)
    ave_ted = sum(total_ted) / len(total_ted)
    return ave_ted


def corpus_edit_distance(preds, refs):
    assert len(preds) == len(refs), 'Length is not same .'
    total_ted = []
    for i in range(len(refs)):
        # list or str object .
        # if type(preds[i]) is list and type(refs[i]) is list:
        #     pred = ' '.join(preds[i])
        #     ref = ' '.join(refs[i])
        if type(preds[i]) is str and type(refs[i]) is str:
            print('Data type is wrong !')
            exit(1)
        ted = edit_distance(preds[i], refs[i])
        if len(refs) <= 20:
            print("Sentence index [%d], Tree edit distance is [%d]" % (i + 1, ted))
        total_ted.append(ted)
    ave_ted = sum(total_ted) / len(total_ted)
    return ave_ted
