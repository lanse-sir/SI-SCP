#from sentence_transformers import SentenceTransformer
#import scipy.spatial as S
import numpy as np
#import distance
import time
import faiss
import argparse

#model = SentenceTransformer('/dfsdata2/yangeg1_data/pretrain-lm/bert-base-nli-mean-tokens')


# def edit_distance(seq1, seq2):
#     return distance.levenshtein(seq1, seq2)


# def ed_matitx(query, corpus):
#     ed = []
#     for j in corpus:
#         query_j = edit_distance(query, j)
#         query_j_norm = query_j / max(len(query), len(j))
#         ed.append(query_j_norm)
#     ed_arr = np.array(ed)
#     return ed_arr


def load_sentence(file_name):
    sentences = []
    with open(file_name, 'r', encoding='utf-8') as f_r:
        for line in f_r:
            sentences.append(line.strip())
    return sentences


def vec_cosine(v1, v2):
    dot_res = np.dot(v1, v2)
    fenmu = np.linalg.norm(v1, ord=2) * np.linalg.norm(v2, ord=2)
    return dot_res / fenmu


def cosine(v, mat):
    sim_list = []
    num = mat.shape[0]
    for i in range(num):
        sim_list.append(vec_cosine(v, mat[i]))
    return np.array(sim_list)


def save_to_file(result, filename):
    with open(filename, 'w', encoding='utf-8') as f_w:
        for original, templates in result:
            f_w.write(original + '\t' + '****' + '\n')
            if templates == []:
                print('No examplar instance !!!')
            else:
                for cos, ed, template in templates:
                    f_w.write(str(cos)[:5] + '\t' + str(ed)[:5] + '\t' + template + '\n')
            f_w.write('\n')


def list_to_file(sents, filename):
    with open(filename, 'w', encoding='utf-8') as f_w:
        for sent in sents:
            f_w.write(sent + '\n')

def main():
    parser = argparse.ArgumentParser(description='Retrieve the most similar sentence as the controlled exemplar.')
    parser.add_argument('-cp',type=str)
    parser.add_argument('-cep',type=str)
    parser.add_argument('-qp', type=str)
    parser.add_argument('-qep',type=str)
    parser.add_argument('-topk', type=int)
    parser.add_argument('-save',type=str)
    args = parser.parse_args()

    m_top = 100
    # corpus_path = 'data/para5m_ref.txt'
    # query_path = 'data/test_src_sent.txt'
    # corpus_emb_fname = 'data/para5m_ref_embedd.npy'
    # querys_emb_fname = 'data/test_src_emb.npy'

    corpus = load_sentence(args.cp)
    querys = load_sentence(args.qp)
    #corpus_pos = load_sentence(cor_pos_fname)
    #querys_pos = load_sentence(querys_pos_fname)
    corpus_embeds = np.load(args.cep)
    querys_embeds = np.load(args.qep)
    dim = corpus_embeds.shape[1]
    print('Corpus embeds size: ', corpus_embeds.shape)
    print('Query embeds size: ', querys_embeds.shape)
    # faiss module .
    index = faiss.IndexFlatL2(dim)
    index.add(corpus_embeds)
    D,I = index.search(querys_embeds, m_top)
    top_index_per_q = I.tolist()
    control_e = []
    for i, index in enumerate(top_index_per_q):
        span_embeds = corpus_embeds[index]
        # distances = scipy.spatial.distance.cdist([querys_embeds[i]], span_embeds, "cosine")[0]
        # cosine_similarity = 1 - distances
        cosine_similarity = cosine(querys_embeds[i], corpus_embeds)
        top_index = cosine_similarity.argsort()[::-1][:args.topk].tolist()
        for j in top_index:
            control_e.append(corpus[j])
    print(len(control_e))
    #save_to_file(res, result_file)
    list_to_file(control_e, args.save)

main()
