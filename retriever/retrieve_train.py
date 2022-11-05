import wandb
import sys
sys.path.append("../")
sys.path.append("../../")

import pickle
from load_file import load_vocab
import argparse

import random
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Dataset import custom_dataset, Data_collator
from Encoders import transformer_encoders
from Optim import ScheduledOptim
from autocg.pretrain_embedding import load_embedding

# torch.multiprocessing.set_start_method("spawn")
# set seed
def seed_everything(seed):
    '''
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def compute_similarity(query_enc, targets_enc):
    # add vector normlize.
    # query_enc = F.normalize(query_enc)
    # targets_enc = F.normalize(targets_enc)

    b_t_sim = torch.matmul(query_enc, targets_enc.transpose(0, 1))
    return b_t_sim


def compute_loss(query_enc, targets_enc, label):
    b_t_sim = compute_similarity(query_enc, targets_enc)
    loss = F.cross_entropy(b_t_sim, label)
    return loss


def evaluate(model, valid_data, k=10):
    valid_nums = 0
    top1_correct = 0
    topk_correct = 0
    model.eval()
    with torch.no_grad():
        for batch in valid_data:
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            sents_enc, parses_enc = model(batch)
            b_t_sim = compute_similarity(sents_enc, parses_enc)
            v, top1 = b_t_sim.topk(1)
            v, topk = b_t_sim.topk(k)
            top1 = top1.tolist()
            topk = topk.tolist()
            labels = batch["label"].tolist()

            valid_nums += b_t_sim.size(0)
            for idx, label in enumerate(labels):
                if label in top1[idx]:
                    top1_correct += 1

                if label in topk[idx]:
                    topk_correct += 1
    top1_acc = top1_correct / valid_nums
    topk_acc = topk_correct / valid_nums
    return top1_acc, topk_acc


def dict_print(d):
    for key, value in d.items():
        print('{0:25}'.format(key), '==> ', value)


def main(args):
    wandb.init(project="retrieve-parse-tree", config=args)

    dict_print(vars(args))

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    # set random seed.
    seed_everything(args.seed)

    # load file, train \ dev
    train = pickle.load(open(args.train, "rb"))
    dev = pickle.load(open(args.dev, "rb"))
    # load word and parse vocab
    word_vocab = load_vocab(args.vocab)
    parse_vocab = load_vocab(args.parse_vocab)

    print("Word Vocab Size: ", len(word_vocab))
    print("Parse Vocab Size: ", len(parse_vocab))

    train_dataset = custom_dataset(train)
    dev_dataset = custom_dataset(dev)
    data_collator = Data_collator(word_vocab, parse_vocab, args)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=data_collator, num_workers=args.num_worker)
    dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=args.batch_size, shuffle=False,
                                collate_fn=data_collator, num_workers=0)

    # load pretrained word embedding.
    if args.pretrain_emb is not None:
        print("Loading pretrained word embedding......")
        word_embedding = load_embedding(args.pretrain_emb, word_vocab)
    else:
        word_embedding = None

    model = transformer_encoders(args, word_vocab, parse_vocab, word_embedding=word_embedding)
    if torch.cuda.is_available():
        model = model.cuda()
    # optimizer inital.
    optimizer = ScheduledOptim(
        torch.optim.Adam(
            model.parameters(),
            betas=(0.9, 0.98), eps=1e-09),
        args.lr, args.d_model, args.warm_step)

    total_loss = 0.
    train_num = 0
    global_steps = 0
    history_scores = []
    for i in range(args.epoch):
        model.train()
        for step, batch in enumerate(train_dataloader):
            global_steps += 1
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}

            sent_enc, parse_enc = model(batch)
            loss = compute_loss(sent_enc, parse_enc, batch["label"])

            optimizer.zero_grad()
            loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step_and_update_lr()
            lr = optimizer._optimizer.param_groups[0]['lr']

            total_loss += loss.item()
            train_num += sent_enc.size(0)
            if global_steps % args.wandb_log == 0:
                wandb.log({'Learning Rate': lr, 'Loss': loss.item()})

            if global_steps % args.log_every == 0:
                print(
                    '[Epoch %d] [Iter %d] Train LOSS=%.5f, Learning ratio=%.5f' % (
                        i, global_steps, total_loss / train_num, lr))

            if global_steps % args.dev_every == 0:
                top1_acc, topk_acc = evaluate(model, dev_dataloader, k=args.topk)
                print(
                    '[Epoch %d] [Iter %d] Validation Set top-1 Acc=%.4f, Top-%d Acc=%.4f' % (
                        i, global_steps, top1_acc, args.topk, topk_acc))

                wandb.log({"Valid Top-1 Acc": top1_acc, "Valid Top-K Acc": topk_acc})
                if history_scores == [] or top1_acc > max(history_scores):
                    history_scores.append(top1_acc)
                    model_file = args.model_file + "_" + str(args.run_name)
                    model.save(model_file)
                    print("Save model to {}".format(model_file))

            model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_name", type=int, default=None)
    parser.add_argument("--train", type=str)
    parser.add_argument("--dev", type=str)
    parser.add_argument("--vocab", type=str)
    parser.add_argument("--parse_vocab", type=str)
    parser.add_argument("--model_file", type=str, default="models/retrieve")
    parser.add_argument("--tree", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_worker", type=int, default=0)
    parser.add_argument("--pretrain_emb", type=str)
    # log print.
    parser.add_argument("--wandb_log", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--dev_every", type=int, default=500)
    parser.add_argument("--topk", type=int, default=10)
    # training config.
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    # parser.add_argument("--eval_bs", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--warm_step", type=int, default=1000)
    parser.add_argument("--sent_enc_layers", type=int, default=4)
    parser.add_argument("--parse_enc_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_seq_len", type=int, default=250)
    parser.add_argument("--embed_size", type=int, default=256)
    parser.add_argument("--head", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--dk", type=int, default=64)
    parser.add_argument("--dv", type=int, default=64)
    parser.add_argument("--d_inner_hid", type=int, default=512)
    args = parser.parse_args()
    main(args)
