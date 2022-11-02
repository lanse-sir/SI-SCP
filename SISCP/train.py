import argparse
import random
import os
import sys
from tqdm import tqdm
import pickle

sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
import math
import wandb
import time

st = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
print("Current time: ", st)
task_name = "Syntax-Controlled Paraphrasing"

import numpy as np
import torch
import torch.nn.functional as F
from syn_control_pg.synpg_transformer import synpg_transformer
from autocg.utils.config_funcs import yaml_to_dict, dict_to_args
# from autocg.optimizers.schedule import LinearWarmupRsqrtDecay
from autocg.pretrain_embedding import load_embedding
from autocg.load_file import save_to_file, load_file_sent_or_parser
# from syn_control_pg.generator import Translator
from syn_control_pg.data_init import batch_tree_encoding
from autocg.evaluation_utils import run_multi_bleu
from syn_control_pg.Optim import ScheduledOptim

# evaluation bleu script .
MULTI_BLEU_PERL = '../eval_tools/multi-bleu.perl'


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training Syntactic Text Generation Models",
        usage="train.py [<args>] [-h | --help]"
    )
    parser.add_argument('--run_name', type=str, help='name of the run.')
    parser.add_argument('--model_config', type=str, help='models configs', default="bpe_15k_paranmt-50w.yaml")
    parser.add_argument('--wandb_every', type=int, default=10)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--dev_every', type=int, default=2000)
    parser.add_argument('--start_eval', type=int, default=-1)

    parser.add_argument('--model_file', type=str, default="cpg.bin")
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--reload', type=str, default=None)
    parser.add_argument('--sent', type=str, default=None)
    parser.add_argument('--beam_size', type=int, default=4)
    opt = parser.parse_args()
    main_args, model_args = None, None

    if opt.model_config is not None:
        model_args = dict_to_args(yaml_to_dict(opt.model_config)['model_configs'])

    return {
        'base': main_args,
        'model': model_args,
        "opt": opt
    }


def load_vocab(fname):
    vocab = {}
    with open(fname, 'r', encoding='utf-8') as f_r:
        for line in f_r:
            if len(line.strip().split()) == 2:
                word, idx = line.strip().split()
                vocab[word] = int(idx)
            if len(line.strip().split()) == 1:
                word = line.strip()
                vocab[word] = len(vocab)
    return vocab


def merge_dict(dict1, dict2):
    res = {**dict1, **dict2}
    return res


def dict_print(d):
    for key, value in d.items():
        print('{0:25}'.format(key), '==> ', value)


def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # sum .
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss


def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


def compute_attn_mse(dec_parse_attns, attns_label, tgt_mask):
    loss = 0.
    # batch_size = attns_label.size(0)
    assert dec_parse_attns[0].size()[-1] == attns_label.size()[-1]
    # print(dec_parse_attns[0].size())
    # print(attns_label.size())
    for layer in dec_parse_attns:
        paths_dim = layer.size()[-1]
        ave_layer = layer.mean(dim=1)
        ave_layer = ave_layer.view(-1, paths_dim) * tgt_mask.view(-1).float().unsqueeze(-1)
        attns_label = attns_label.view(-1, paths_dim)
        attn_loss = F.mse_loss(ave_layer, attns_label, reduction='sum')

        loss += attn_loss
    ave_loss = loss / len(dec_parse_attns)
    return ave_loss


def eval_ppl(model, test_data, word_vocab, parse_vocab, opt, pos_vocab, parent_child):
    batch_size = opt.eval_bs

    model.eval()
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i + batch_size]
            seq_tensor = batch_tree_encoding(batch, word_vocab, parse_vocab, pos_vocab=pos_vocab, opt=opt,
                                             parent_child=parent_child,
                                             if_train=True)
            pred, _ = model(seq_tensor)
            loss, n_correct, n_word = cal_performance(pred, seq_tensor['tgt_sent'], word_vocab['<PAD>'],
                                                      smoothing=True)
            # compute ppl .
            total_loss += loss.item()
            n_word_total += n_word
            n_word_correct += n_correct

    loss_per_word = total_loss / n_word_total
    accuracy = n_word_correct / n_word_total
    return loss_per_word, accuracy


def anneal_function(i, x0, init_lr, lam):
    if i > x0:
        lr = init_lr * lam ** (i - x0)
        return lr
    else:
        return init_lr


def main(config_args):
    opt = config_args['opt']
    model_args = config_args['model']
    # merge the base args and model args.
    args_dict = merge_dict(vars(opt), vars(model_args))
    opt = argparse.Namespace(**args_dict)
    
    # setup wandb.
    wandb.init(project=task_name, name=opt.run_name)
    # set up initial seed .
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    dict_print(args_dict)
    print('-' * 120)
    train_sent = load_file_sent_or_parser(opt.src, 'sent')
    train_parse = load_file_sent_or_parser(opt.ref_parse, 'parse')
    train_ref = load_file_sent_or_parser(opt.ref, 'sent')

    valid_sent = load_file_sent_or_parser(opt.dev_src, 'sent')
    valid_parse = load_file_sent_or_parser(opt.dev_parse, 'parse')
    valid_ref = load_file_sent_or_parser(opt.dev_ref, 'sent')

    train_data = list(zip(train_sent, train_parse, train_ref))
    valid_data = list(zip(valid_sent, valid_parse, valid_ref))
    print('All train instance is %d, dev instance is %d' % (len(train_data), len(valid_data)))
    print(train_data[0][0])
    print(train_data[0][1])
    print(train_data[0][2])

    vocab = load_vocab(opt.vocab)
    parse_vocab = load_vocab(opt.parser_vocab)

    pos_vocab = load_vocab(opt.pos_vocab)
    print("Vocab Size :", len(vocab))

    # parent_child = pickle.load(open(opt.parent_child, "rb"))
    parent_child = None
    # load pretrained word embedding.
    # if os.path.exists(opt.pretrain_emb) and opt.debug is not True:
        # word_embedding = load_embedding(opt.pretrain_emb, vocab)
    # else:
        # word_embedding = None
    
    word_embedding=None
    if opt.pretrained is not None:
        print(f'Loading Pretrained Model {opt.pretrained}')
        params = torch.load(opt.pretrained, map_location=lambda storage, loc: storage)
        model = synpg_transformer(params['args'], params['word_vocab'], params['parse_vocab'])
        model.load_state_dict(params['state_dict'])
    else:
        model = synpg_transformer(opt, vocab, parse_vocab, word_embedding=word_embedding)

    if opt.cuda and torch.cuda.is_available():
        # set up use GPU id
        torch.cuda.set_device(opt.gpu)

        model = model.cuda()

    # optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)

    optimizer = ScheduledOptim(
        torch.optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        opt.lr, opt.d_model, opt.warm_step)

    # hyper parameter .

    # translator = Translator(model=model, beam_size=opt.beam_size, max_seq_len=opt.sent_max_time_step)

    batch_size = opt.batch_size

    # start to train .
    step = 0
    train_num = 0
    total_loss = 0.0
    total_reconstruct_loss = 0.
    total_attn_loss = 0.
    n_word_total, n_word_correct = 0, 0
    history_scores = []
    for i in range(opt.epoch):
        random.shuffle(train_data)
        # word_dropout = anneal_function(i + 1, x0=opt.x0, init_lr=1.0, lam=opt.lamda)
        # model.args.word_drop = word_dropout
        # print('[Epoch %d], Dropout rate=%.3f' % (i + 1, model.args.word_drop))
        for idx in range(0, len(train_data), batch_size):
            # set training mode .
            batch = train_data[idx:idx + batch_size]
            # set up learning rate .
            step += 1
            train_num += len(batch)
            # if opt.linear_warmup:
            #    lr = LinearWarmupRsqrtDecay(warm_up=opt.warm_up, init_lr=0.0, max_lr=opt.lr, step=step)
            #    optimizer.param_groups[0]['lr'] = lr
            # ... generate batch tensor .
            seq_tensor = batch_tree_encoding(batch, vocab, parse_vocab, opt, pos_vocab=pos_vocab,
                                             parent_child=parent_child,
                                             if_train=True,
                                             node_noise=opt.node_noise)
            #pickle.dump(seq_tensor, open("qqp/tensor.pkl", "wb"))
            #exit()
            
            pred, dec_parse_attns = model(seq_tensor)
            loss, n_correct, n_word = cal_performance(pred, seq_tensor['tgt_sent'], vocab['<PAD>'],
                                                      smoothing=True)

            syn_attn_mask = seq_tensor['tgt_mask'].float() * seq_tensor['drop_mask'].float()
            attn_loss = compute_attn_mse(dec_parse_attns, seq_tensor['attn_label'], syn_attn_mask)

            total_loss += loss.item() + attn_loss.item()
            total_reconstruct_loss += loss.item()
            total_attn_loss += attn_loss.item()
            n_word_correct += n_correct
            n_word_total += n_word
            optimizer.zero_grad()
            (opt.mul_sem * loss + opt.mul_attn * attn_loss).backward()

            # if opt.clip > 0:
            #    torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
            optimizer.step_and_update_lr()

            if step % opt.wandb_every == 0:
                wandb.log({"train loss": total_loss / train_num, "sentence loss": loss.item() / len(batch),
                           "attn loss": attn_loss.item()})

            if step % opt.log_every == 0:
                print(
                    '[Epoch %d] [Iter %d] Train LOSS=%.3f, Sentence Loss=%.3f, Attn Loss=%.3f' % (
                        i, step, total_loss / train_num, total_reconstruct_loss / train_num,
                        total_attn_loss / train_num))

            if i > opt.start_eval and step % opt.dev_every == 0:
                valid_loss, valid_word_acc = eval_ppl(model=model,
                                                      test_data=valid_data,
                                                      word_vocab=vocab,
                                                      parse_vocab=parse_vocab,
                                                      opt=opt,
                                                      pos_vocab=pos_vocab,
                                                      parent_child=parent_child)
                valid_ppl = math.exp(min(valid_loss, 100))
                # print('[Epoch %d] [Iter %d] Validation PPL is %.4f' % (i, step, valid_ppl))
                print('[Epoch %d] [Iter %d] Dev ppl=%.3f, Dev word accuracy=%.3f' % (
                    i, step, valid_ppl, valid_word_acc))
                wandb.log({"valid ppl": valid_ppl, "valid word accuracy": valid_word_acc})

                # metric = -valid_ppl
                metric = valid_word_acc
                is_better = (history_scores == []) or metric > max(history_scores)
                if is_better:
                    history_scores.append(metric)
                    model_file = os.path.join("models", st, opt.model_file)
                    model.save(model_file)
                    print('save model to [%s]' % model_file, file=sys.stdout)
                    # m, optimizer, patience = get_lr_schedule(is_better, model, optimizer, opt, patience, model_file)
                model.train()


def auto_evaluate(model, test_data, word_vocab, parse_vocab, pos_vocab, opt, parent_child=None):
    preds = []
    batch_size = opt.eval_bs

    attns = []
    leaf_nodes = []
    with torch.no_grad():
        for i in tqdm(range(0, len(test_data), batch_size)):
            batch = test_data[i:i + batch_size]
            input_tensors = batch_tree_encoding(batch, word_vocab, parse_vocab, pos_vocab=pos_vocab, opt=opt,
                                                parent_child=parent_child,
                                                if_train=False)
            if opt.greedy:
                sent_results, syn_attns = model.greedy_decode(input_tensors, return_attns=opt.return_attn)
                print(len(syn_attns))
                attns += syn_attns
                leaf_nodes += input_tensors["leaf"]
            else:
                sent_results, _ = model.translate_sentences(input_tensors)
            preds += sent_results

    filename = opt.sent
    if opt.return_attn:
        node_words = (preds, leaf_nodes, attns)
        npy_file = filename + ".pkl"
        print(f"Save leaf & attention weight to {npy_file}.")
        with open(npy_file, "wb") as fw:
            pickle.dump(node_words, fw)

    print('Save generate sentence to {} .'.format(filename))
    save_to_file(preds, filename)

    ori_bleu = -1.0
    ref_bleu = -1.0
    if opt.dev_src != '':
        ori_bleu = run_multi_bleu(filename, opt.dev_src, MULTI_BLEU_PERL)
    if opt.dev_ref != '':
        ref_bleu = run_multi_bleu(filename, opt.dev_ref, MULTI_BLEU_PERL)
    return ori_bleu, ref_bleu


if __name__ == "__main__":
    config_args = parse_args()
    # args = config_args['opt']
    main(config_args)
