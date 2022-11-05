import torch
from torch.utils.data import Dataset
from data_collator import trees_to_dict, data_to_idx


class custom_dataset(Dataset):
    def __init__(self, data):
        super(custom_dataset, self).__init__()
        self.src_text = data["src_sents"]
        self.src_pos = data["src_pos"]

        self.tgt_parse = data["tgt_parse"]

        print("Dataset Size: {}".format(len(self.src_text)))
        # print("Linear parse tree convert to dict......")
        # self.tgt_parse = trees_to_dict(data["tgt_parse"])

    def __len__(self):
        return len(self.src_text)

    def __getitem__(self, item):
        return self.src_text[item], self.src_pos[item], self.tgt_parse[item]


class Data_collator:
    def __init__(self, word_vocab, parse_vocab, args):
        self.word_vocab = word_vocab
        self.parse_vocab = parse_vocab
        self.args = args

    def __call__(self, batch):
        return self.collator_fn(batch)

    def batch_process(self, seqs, vocab):
        sents = [["<CLS>"] + seq for seq in seqs]
        sents_idx_seq = [data_to_idx(s, vocab) for s in sents]
        seq_lengths, seq_tensor, seq_mask = self.padding(sents_idx_seq)
        return seq_lengths, seq_tensor, seq_mask

    def padding(self, seq_ids):
        batch_size = len(seq_ids)
        word_seq_lengths = torch.LongTensor(list(map(len, seq_ids)))
        # print(word_seq_lengths)
        max_seq_len = word_seq_lengths.max().item()
        # max_seq_len = opt.sent_max_time_step
        word_seq_tensor = torch.zeros((batch_size, max_seq_len)).long()
        # tgt_word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
        mask = torch.zeros((batch_size, max_seq_len)).byte()
        for idx, (seq, seqlen) in enumerate(zip(seq_ids, word_seq_lengths)):
            seqlen = seqlen.item()
            word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
            mask[idx, :seqlen] = torch.Tensor([1] * seqlen)
        return word_seq_lengths, word_seq_tensor, mask

    def collator_fn(self, batch):
        batch_size = len(batch)

        # sentence process.
        sents = [["<CLS>"] + pair[0] for pair in batch]
        sents_idx_seq = [data_to_idx(s, self.word_vocab) for s in sents]
        src_seq_lengths, src_seq_tensor, src_mask = self.padding(sents_idx_seq)
        # part-of-speech process.
        pos = [["<CLS>"] + pair[1] for pair in batch]
        pos_idx_seq = [data_to_idx(p, self.parse_vocab) for p in pos]
        src_pos_seq_lengths, src_pos_seq_tensork, src_pos_mask = self.padding(pos_idx_seq)

        def dfs_travel(nodes, tree, root, stack, lj_matrix, leave_node, level, height):
            # nodes.append(root.rsplit('-', maxsplit=1)[0])
            nodes.append(root)
            level.append(height)

            stack.append(root)

            if len(tree[root]) != 0:
                for child in tree[root]:
                    lj_matrix.append((stack[-1], child))  # tree attention.
                    dfs_travel(nodes, tree, child, stack, lj_matrix, leave_node, level, height + 1)
                stack.pop()
            else:
                leave_node.append(len(nodes) - 1)
                stack.pop()
                # path attention.
                # for parient in stack:
                #     lj_matrix.append((parient, root))

        def deep_travel_node(parses):
            nodes = []
            ljs = []
            leave_nodes = []
            levels = []
            for parse in parses:
                stack = []
                lj = []
                node = []
                leave_node = []
                level = []
                dfs_travel(node, parse, 'ROOT', stack, lj, leave_node, level, height=1)
                # paths_every_batch.append([path.split('->') for path in paths_every_tree])
                nodes.append(node)
                ljs.append(lj)
                leave_nodes.append(leave_node)
                levels.append(level)
            return nodes, levels, ljs, leave_nodes

        # parse tree process.
        parses = [pair[2] for pair in batch]
        tgt_parse_set = list(set(parses))

        target_label = torch.LongTensor([tgt_parse_set.index(parse) for parse in parses])

        if self.args.tree:
            tgt_trees = trees_to_dict(tgt_parse_set)
            nodes, levels, ljs, leave_nodes = deep_travel_node(tgt_trees)
            max_node_seq_len = max(list(map(len, nodes)))
            tgt_trees_attn_mask = torch.eye(max_node_seq_len).byte()
            tgt_trees_attn_mask = tgt_trees_attn_mask.expand(batch_size, max_node_seq_len, max_node_seq_len)

            for idx, lj in enumerate(ljs):
                for head, tail in lj:
                    row, column = nodes[idx].index(head), nodes[idx].index(tail)
                    tgt_trees_attn_mask[idx, row, column] = 1
                    tgt_trees_attn_mask[idx, column, row] = 1  # syntemtric. duichen

            tgt_nodes_seq = [data_to_idx([n.rsplit('-', maxsplit=1)[0] for n in seq], self.parse_vocab) for seq in
                             nodes]

            _, tgt_nodes_tensor, _ = self.padding(tgt_nodes_seq)
            _, levels_seq_tensor, _ = self.padding(levels)
            _, leave_nodes_tensor, leave_nodes_mask = self.padding(leave_nodes)

            return {
                "src_lengths": src_seq_lengths,
                "src_seq": src_seq_tensor,
                "src_mask": src_mask,

                "src_pos_lengths": src_pos_seq_lengths,
                "src_pos_seq": src_pos_seq_tensork,
                "src_pos_mask": src_pos_mask,

                "tree_tensor": tgt_nodes_tensor,
                "tree_mask": tgt_trees_attn_mask,
                "tree_height": levels_seq_tensor,

                "label": target_label

            }

        else:
            tgt_trees = [["<CLS>"] + p.split() for p in tgt_parse_set]
            tgt_idx_seq = [data_to_idx(p, self.parse_vocab) for p in tgt_trees]
            tgt_seq_lengths, tgt_seq_tensor, tgt_mask = self.padding(tgt_idx_seq)

            return {
                "src_lengths": src_seq_lengths,
                "src_seq": src_seq_tensor,
                "src_mask": src_mask,

                "src_pos_lengths": src_pos_seq_lengths,
                "src_pos_seq": src_pos_seq_tensork,
                "src_pos_mask": src_pos_mask,

                "tree_tensor": tgt_seq_tensor,
                "tree_mask": tgt_mask,

                "label": target_label
            }
