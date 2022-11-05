from load_file import load_file_sent_or_parser, save_to_file
from Dataset import Data_collator
from Encoders import transformer_encoders
from sentence_transformers import util
from retrieve_train import seed_everything, compute_similarity
import torch
import argparse


# def to_cuda(**kwargs):
#     if torch.cuda.is_available():


def main(args):
    # seed_everything(0)
    # Load data.
    query_sents = load_file_sent_or_parser(args.input_sent)
    query_pos = load_file_sent_or_parser(args.input_pos)
    tgt_parses = load_file_sent_or_parser(args.tgt_parse, type="parse")
    corpus = load_file_sent_or_parser(args.corpus, type="parse")

    labels = [corpus.index(p) if p in corpus else "N" for p in tgt_parses]

    # split process.
    corpus = [_.split() for _ in corpus]

    # Set up GPU environment and load model.
    cuda = True if torch.cuda.is_available() else False
    torch.cuda.set_device(args.gpu)
    params = torch.load(args.model_file, map_location=lambda storage, loc: storage)
    model_args = params['args']
    word_vocab = params['word_vocab']
    parse_vocab = params['parse_vocab']

    encoders = transformer_encoders(model_args, word_vocab, parse_vocab)
    encoders.load_state_dict(params['state_dict'])
    encoders.eval()  # importance !!!
    if torch.cuda.is_available():
        encoders = encoders.cuda()

    # batch preprocess ......
    data_collator = Data_collator(word_vocab=word_vocab, parse_vocab=parse_vocab, args=args)

    #  corpus encode.
    with torch.no_grad():
        print("Corpus Encoding ......")
        corpus_embeddings = []
        for start in range(0, len(corpus), args.batch_size):
            tgt_trees = corpus[start:start + args.batch_size]
            lengths, tensor, mask = data_collator.batch_process(tgt_trees, vocab=parse_vocab)
            mask = mask.unsqueeze(-2)
            if cuda:
                lengths, tensor, mask = lengths.cuda(), tensor.cuda(), mask.cuda()

            tree_embeds = encoders.corpus_encode(tensor, mask)
            corpus_embeddings.append(tree_embeds)
        corpus_embeddings = torch.cat(corpus_embeddings)

        # query encode.
        assert len(query_sents) == len(query_pos)
        print("Query Encoding ......")
        query_embeddings = []
        for q_start in range(0, len(query_sents), args.batch_size):
            sentences = query_sents[q_start:q_start + args.batch_size]
            sent_lengths, sent_tensor, sent_mask = data_collator.batch_process(sentences, vocab=word_vocab)
            sent_mask = sent_mask.unsqueeze(-2)

            poses = query_pos[q_start:q_start + args.batch_size]
            pos_lengths, pos_tensor, pos_mask = data_collator.batch_process(poses, vocab=parse_vocab)
            pos_mask = pos_mask.unsqueeze(-2)
            if cuda:
                sent_lengths, sent_tensor, sent_mask = sent_lengths.cuda(), sent_tensor.cuda(), sent_mask.cuda()
                pos_lengths, pos_tensor, pos_mask = pos_lengths.cuda(), pos_tensor.cuda(), pos_mask.cuda()

            query_embed = encoders.query_encode(sent_tensor, sent_mask, pos_tensor, pos_mask)
            query_embeddings.append(query_embed)
        query_embeddings = torch.cat(query_embeddings)

    # hist = util.semantic_search(query_embeddings, corpus_embeddings, top_k=args.k, score_function=util.dot_score)
    # print(query_embeddings[0])
    q_c_sim = compute_similarity(query_embeddings, corpus_embeddings)

    print("Total Nums: ", q_c_sim.size(0))
    # scores = []
    # ids = []
    # for item in hist:
    #     scores.append([x["score"] for x in item])
    #     ids.append([x["corpus_id"] for x in item])
    scores, ids = q_c_sim.topk(args.k)
    ids = ids.tolist()
    scores = scores.tolist()
    print(ids[0])
    print(scores[0])
    r1 = 0
    r5 = 0
    r10 = 0
    for i in range(len(labels)):
        if labels[i] in ids[i][:1]:
            r1 += 1

        if labels[i] in ids[i][:5]:
            r5 += 1

        if labels[i] in ids[i][:args.k]:
            r10 += 1
    print("R@1: {}, R@5: {}, R@{}: {}".format(r1 / len(labels), r5 / len(labels), args.k, r10 / len(labels)))
    print("Source Sentence: ", " ".join(query_sents[args.idx]))
    print("Source Parse: ", ' '.join(query_pos[args.idx]))
    print("Gold Parse: ", tgt_parses[args.idx])
    print("Retrieve Parse: ")
    for id in ids[args.idx]:
        print(" ".join(corpus[id]))

    lines = []
    for mul_ids in ids:
        lines.append("\n".join([" ".join(corpus[i]) for i in mul_ids]))
    if args.out is not None:
        print(f"Save to {args.out}")
        save_to_file(lines, args.out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model_file", type=str)
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--input_sent", type=str)
    parser.add_argument("--input_pos", type=str)
    parser.add_argument("--tgt_parse", type=str)

    parser.add_argument("--corpus", type=str)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    main(args)
