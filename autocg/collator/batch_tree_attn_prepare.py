import torch
import json


def batch_serialization_parallel(batch, word_vocab, parse_vocab, opt, pos_vocab=None, word_occurence=None,
                                 document_num=None,
                                 if_train=True):
    batch_size = len(batch)

    def padding(sents):
        word_seq_lengths = torch.LongTensor(list(map(len, sents)))
        # print(word_seq_lengths)
        # max_seq_len = word_seq_lengths.max().item()
        max_seq_len = opt.sent_max_time_step
        word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
        # tgt_word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
        mask = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).byte()
        for idx, (seq, seqlen) in enumerate(zip(sents, word_seq_lengths)):
            seqlen = seqlen.item()
            word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
            mask[idx, :seqlen] = torch.Tensor([1] * seqlen)
        return word_seq_lengths, word_seq_tensor, mask

    def att_to_onehot(atts):
        lengths = torch.LongTensor(list(map(len, atts)))
        # max_seq_len = lengths.max().item()

        max_seq_len = opt.sent_max_time_step
        leave_lengths = [seq[-1] for seq in atts]
        max_leaves = max(leave_lengths)
        one_hot = torch.zeros((batch_size, max_seq_len, max_leaves))
        label_smoothing = 0.1
        for b in range(batch_size):
            lens = lengths[b].item()
            assert leave_lengths[b] != 0  # ensure the fenmu is not zero.
            one_hot[b, :lens, :leave_lengths[b]] = torch.FloatTensor(
                [label_smoothing / leave_lengths[b]] * leave_lengths[b])
            one_hot[b, :lens].scatter_add_(-1, (torch.LongTensor(atts[b]) - 1).unsqueeze(1),
                                           torch.FloatTensor([1 - label_smoothing] * len(atts[b])).unsqueeze(1))
        return one_hot

    # sentence process.
    sents = [pair[0] for pair in batch]
    sents_idx_seq = [data_to_idx(s, word_vocab) for s in sents]

    src_seq_lengths, src_seq_tensor, src_mask = padding(sents_idx_seq)

    # parse tree process.
    # string sequence.
    # if opt.tree_gru:
    parsers = [json.loads(pair[1]) for pair in batch]

    if opt.remove_pos:
        remove_pos_from_trees(parsers, pos_vocab)
    remove_leaves_from_trees(parsers, bpe=opt.bpe, pos=opt.remove_pos)
    tgt_trees, tgt_phrase_starts = trim_trees(parsers, opt.ht)
    paths_every_batch = trees_to_paths(parsers)

    paths_nums = torch.LongTensor(list(map(len, paths_every_batch)))
    max_path_nums = paths_nums.max().item()
    tree_tensor = torch.zeros(batch_size, max_path_nums, opt.ht).long()
    tree_mask = torch.zeros(batch_size, max_path_nums, opt.ht).byte()
    tree_height_tensor = torch.ones(batch_size, max_path_nums).long()
    tree_paths_mask = torch.zeros(batch_size, max_path_nums).byte()
    for i in range(batch_size):
        paths_every_sent = paths_every_batch[i]
        path_nums_every_sent = len(paths_every_sent)
        tree_paths_mask[i, :path_nums_every_sent] = torch.Tensor([1] * path_nums_every_sent)

        # # #
        for j in range(len(paths_every_sent)):
            deep = len(paths_every_sent[j])
            node_seq = data_to_idx(paths_every_sent[j], parse_vocab)
            tree_tensor[i, j, :deep] = torch.LongTensor(node_seq)
            tree_mask[i, j, :deep] = torch.Tensor([1] * deep)
            tree_height_tensor[i, j] = deep
    # else:
    #     parsers = [pair[1] for pair in batch]
    #     parsers_idx_seq = [data_to_idx(parse, parse_vocab) for parse in parsers]
    #
    #     src_parse_seq_lengths, src_parse_seq_tensor, src_parse_mask = padding(parsers_idx_seq)
    #     src_parse_seq_lengths, src_parse_perm_idx = src_parse_seq_lengths.sort(0, descending=True)
    #     src_parse_seq_tensor = src_parse_seq_tensor[src_parse_perm_idx]
    #     src_parse_mask = src_parse_mask[src_parse_perm_idx]
    #     _, parse_seq_recover = src_parse_perm_idx.sort(0, descending=False)

    # class parse tree: classifer to decode the controlled syntax.
    # if opt.tree_gru:
    #     c_parse_seq = trees_travel(dict_trees, root='ROOT')
    # c_parse_seq = [pair[1].split() for pair in batch]
    # else:
    #     c_parse_seq = parsers
    # c_tgt_parses = [['<s>'] + seq + ['</s>'] for seq in c_parse_seq]
    # c_tgt_parses_idx = [data_to_idx(parse, parse_vocab) for parse in c_tgt_parses]
    # c_tgt_parses_lengths, c_tgt_parses_seq, c_tgt_parses_mask = padding(c_tgt_parses_idx)

    # Target sentence bag of word label .
    # contents = select_content_words(sents, word_occurence, document_num, ratio=opt.bow_ratio)
    # contents_id = [[word_vocab[w] for w in sent if w in word_vocab] for sent in contents]
    # _, contents_seq_tensor, _ = padding(contents_id)

    # assert len(contents_id) == batch_size
    # bag_label = torch.zeros((batch_size, len(word_vocab))).float()
    # for idx in range(batch_size):
    #     bag_label[idx].scatter_(-1, torch.LongTensor(contents_id[idx]), 1.)

    if if_train:

        # tgt_sents = [['<s>'] + pair[2] + ['</s>'] for pair in batch]
        inps = [['<s>'] + pair[2] for pair in batch]
        tgts = [pair[2] + ['</s>'] for pair in batch]
        # tgt_parsers = [['<s>'] + pair[2] + ['</s>'] for pair in batch]
        # tgt_parsers_idx_seq = [data_to_idx(parse, parse_vocab) for parse in tgt_parsers]
        tgt_idx_seq = [data_to_idx(tgt_s, word_vocab) for tgt_s in tgts]
        inp_idx_seq = [data_to_idx(inp, word_vocab) for inp in inps]
        # tgt_parse_seq_lengths, tgt_parse_seq_tensor, tgt_parse_mask = padding(tgt_parsers_idx_seq)

        inp_seq_lengths, inp_seq_tensor, inp_mask = padding(inp_idx_seq)
        tgt_seq_lengths, tgt_seq_tensor, tgt_mask = padding(tgt_idx_seq)

        att_targets = []
        for i in range(batch_size):
            flag = 0
            t = []
            for j in range(min(len(tgt_idx_seq[i]), opt.sent_max_time_step)):
                if j in tgt_phrase_starts[i]:
                    flag += 1
                t.append(flag)
            att_targets.append(t)
        smooth_one_hot = att_to_onehot(att_targets)
    else:
        tgt_seq_tensor = torch.Tensor([0.])
        inp_seq_tensor = torch.Tensor([0.])
        inp_mask = torch.Tensor([0.])
        smooth_one_hot = torch.Tensor([0.])
        tgt_mask = torch.Tensor([0.])

    if torch.cuda.is_available():
        return {
            'src_lengths': src_seq_lengths.cuda(),
            'src_seq': src_seq_tensor.cuda(),
            'src_mask': src_mask.cuda(),
            'tree_tensor': tree_tensor.cuda(),
            'tree_mask': tree_mask.cuda(),
            'tree_height': tree_height_tensor.cuda(),
            'tree_path_mask': tree_paths_mask.cuda(),
            'inp_seq': inp_seq_tensor.cuda(),
            'inp_mask': inp_mask.cuda(),
            'tgt_sent': tgt_seq_tensor.cuda(),
            'attn_label': smooth_one_hot.cuda(),
            'tgt_mask': tgt_mask.cuda()
        }
    else:
        return {
            'src_lengths': src_seq_lengths,
            'src_seq': src_seq_tensor,
            'src_mask': src_mask,
            'tree_tensor': tree_tensor,
            'tree_mask': tree_mask,
            'tree_height': tree_height_tensor,
            'tree_path_mask': tree_paths_mask,
            'inp_seq': inp_seq_tensor,
            'inp_mask': inp_mask,
            'tgt_sent': tgt_seq_tensor,
            'attn_label': smooth_one_hot,
            'tgt_mask': tgt_mask
        }
