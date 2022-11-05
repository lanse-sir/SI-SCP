import sys
sys.path.append("../")
sys.path.append("../../")

import argparse
import os
from nltk.tree import Tree
from autocg.load_file import load_sentence, load_vocab, save_to_file
from tqdm import tqdm


def cleanbrackets(string):
    a = {
        '-LRB-': '(',
        '-RRB-': ')',
        '-LSB-': '[',
        '-RSB-': ']',
        '-LCB-': '{',
        '-RCB-': '}',

        '-lrb-': '(',
        '-rrb-': ')',
        '-lsb-': '[',
        '-rsb-': ']',
        '-lcb-': '{',
        '-rcb-': '}',
    }
    try:
        return a[string]
    except:
        return string


def dfs(root, s, bpe=False):
    # label = root._label.split()[0].replace('-','')
    label = root.label()
    # label = re.findall(r'-+[A-Z]+-|[A-Z]+\$|[A-Z]+|\.', root._label)[0]
    tree_dict = {label: []}
    # print(leaf, root._label)
    # if len(root._label.split()) > 1:
    #     tree_dict[label].append(root._label.split()[1])

    for child in root:
        if type(child) is str:
            if bpe:
                tree_dict[label] = cleanbrackets(child)
            else:
                tree_dict[label] = cleanbrackets(child)
            return tree_dict
        # if child._label.split()[]
        else:
            tree_dict[label].append(dfs(child, s, bpe))
    return tree_dict


def dfsopti(tree, root, tree_dict, label, cnt):
    tree_dict[label] = []

    if type(tree[root]) is str:
        tree_dict[label].append(tree[root])
    else:
        for child in tree[root]:
            if type(child) is dict:
                child_label = list(child.keys())[0]
                if child_label not in cnt:
                    cnt[child_label] = 0
                cnt[child_label] += 1
                child_label = child_label + '-' + str(cnt[child_label])
                tree_dict[label].append(child_label)
                dfsopti(child, list(child.keys())[0], tree_dict, child_label, cnt)


def remove_pos_from_tree(tree, root, cnt, ht, pos_tags):
    words = []
    child_list = []
    for idx, child in enumerate(tree[root]):
        if child in tree:
            if ht > 2:
                if child.rsplit('-', maxsplit=1)[0] in pos_tags:
                    # remove relative node.
                    words.append(tree[child][0])

                    tree.pop(child)
                    tree['<POS>' + '-' + str(cnt['<POS>'])] = words

                    node_label = '<POS>' + '-' + str(cnt['<POS>'])
                    if node_label not in child_list:
                        child_list.append(node_label)

                else:
                    child_list.append(child)
                    if len(words) != 0:
                        cnt['<POS>'] += 1
                        words = []

        # at first has pop operation .
        if child in tree:
            # if len(words) != 0:
            #     cnt['<POS>'] += 1
            remove_pos_from_tree(tree, child, cnt, ht + 1, pos_tags)

    if ht > 2:
        if len(child_list) != 0:
            # child_list.insert(0, '<POS>' + '-' + str(cnt['<POS>']))
            # tree['<POS>' + '-' + str(cnt['<POS>'])] = words
            tree[root] = child_list
            cnt['<POS>'] += 1


def tree_to_string(tree, root, node_list):
    node_list.append('(')
    node_list.append(root.rsplit('-', maxsplit=1)[0])
    # node_list.append(')')
    for child in tree[root]:
        if child in tree:
            tree_to_string(tree, child, node_list)

        # need word or not.
        # else:
        #     node_list.append(child)

        # node_list.append(')')
    node_list.append(')')


def main_remove_pos(parse, pos_tags):
    p_tree = Tree.fromstring(parse)
    p_dict = dfs(p_tree, '', False)
    tree = p_dict
    tree_dict = {}
    cnt = {'ROOT': 0}
    dfsopti(tree, 'ROOT', tree_dict, 'ROOT', cnt)

    remove_pos_from_tree(tree_dict, 'ROOT', cnt={'<POS>': 1}, ht=1, pos_tags=pos_tags)

    list_tree = list()

    tree_to_string(tree_dict, "ROOT", list_tree)

    string_tree = ' '.join(list_tree)
    return string_tree


def mains_remove_pos(parses, pos_tags):
    re = []
    for i in tqdm(range(len(parses))):
        re.append(main_remove_pos(parses[i], pos_tags))
    return re


def main(args):
    src_file = os.path.join(args.root, args.dir, f'{args.dir}_src_parse_tree.txt')
    tgt_file = os.path.join(args.root, args.dir, f'{args.dir}_tgt_parse_tree.txt')

    print(src_file)
    print(tgt_file)
    src_parses = load_sentence(src_file)
    tgt_parses = load_sentence(tgt_file)

    pos_tags = load_vocab(args.pos_file)

    src_remove_pos = mains_remove_pos(src_parses, pos_tags)
    print('')
    tgt_remove_pos = mains_remove_pos(tgt_parses, pos_tags)
    print('')

    src_rpos_file = os.path.join(args.root, args.dir, f'{args.dir}_src_parse_tree_remove-pos.txt')
    tgt_rpos_file = os.path.join(args.root, args.dir, f'{args.dir}_tgt_parse_tree_remove-pos.txt')
    print(f"Save to {src_rpos_file}. ")
    print(f"Save to {tgt_rpos_file}. ")

    save_to_file(src_remove_pos, src_rpos_file)
    save_to_file(tgt_remove_pos, tgt_rpos_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data/paranmt-350k-4~30')
    parser.add_argument('--dir', type=str, default='train')
    parser.add_argument('--pos_file', type=str)
    args = parser.parse_args()
    main(args)
