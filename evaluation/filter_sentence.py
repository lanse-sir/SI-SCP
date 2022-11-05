from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from eval_utils import compute_self_bleu, compute_bleu
import numpy as np
import argparse
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction, sentence_bleu
cherry = SmoothingFunction()

model_path = '../eval_tools/distilroberta-base-paraphrase-v1'
model = SentenceTransformer(model_path)


def multi_save(srcs, preds, filename):
    assert len(srcs)==len(preds)
    
    with open(filename, "w", encoding="utf-8") as fw:
        for line in range(len(srcs)):
            fw.write("Source: "+srcs[line]+"\n")
            for p in preds[line]:
                fw.write(p+"\n")
            fw.write("\n")
    print(f"Save to {filename}")
        
def load_sentence(file_name):
    sentences = []
    with open(file_name, 'r', encoding='utf-8') as f_r:
        for line in f_r:
            sentences.append(line.strip())
    return sentences

def select_bleu(src, candidates):
    max_bleu = -1
    max_idx = None
    for i, cand in enumerate(candidates):
        bleu_score = sentence_bleu([src], cand, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=cherry.method1)
        
        if bleu_score > max_bleu:
            max_bleu = bleu_score
            max_idx = i
    return max_idx


def compute_bleu_1vn(src, candidates):
    b1_l=[]
    b2_l=[]
    b3_l=[]
    b4_l=[]
    for i, cand in enumerate(candidates):
        b1 = sentence_bleu([src], cand, weights=(1.0, 0., 0., 0.), smoothing_function=cherry.method1)
        b2 = sentence_bleu([src], cand, weights=(0.5, 0.5, 0., 0.), smoothing_function=cherry.method1)
        b3 = sentence_bleu([src], cand, weights=(0.33, 0.33, 0.34, 0), smoothing_function=cherry.method1)
        b4 = sentence_bleu([src], cand, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=cherry.method1)
        
        b1_l.append(b1)
        b2_l.append(b2)
        b3_l.append(b3)
        b4_l.append(b4)    
    return b1_l, b2_l, b3_l, b4_l

def pprint_min_max_ave(ll):
    print("Min: ", min(ll))
    print("Max: ", max(ll))
    print("Ave: ", sum(ll)/len(ll))
    

def main(args):
    preds = load_sentence(args.pred)
    srcs = load_sentence(args.src)
    refs = load_sentence(args.ref)
    corpus_embeds = model.encode(preds, batch_size=64)
    srcs_embeds = model.encode(srcs, batch_size=64)
    print(corpus_embeds.shape)
    print(srcs_embeds.shape)
    
    n = corpus_embeds.shape[0]//srcs_embeds.shape[0]
    print("N: ", n)
    #all_sim = []
    
    all_b1 = []
    all_b2 = []
    all_b3 = []
    all_b4 = []
    
    remain = []
    remain_num = []
    remain_sim = []
    all_sim = []
    
    ref_best_bleu = []
    src_best_bleu = []
    sbert_best_bleu = []
    for i in tqdm(range(len(srcs))):
        
        remain_pred = []
        cor_preds = preds[i*n:(i+1)*n]
        cor_embeds = corpus_embeds[i*n:(i+1)*n]
        
        # print(cor_preds)
        # select best bleu with reference.
        ref_max_idx = select_bleu(refs[i], cor_preds)
        ref_best_bleu.append(cor_preds[ref_max_idx])
        
        # select best bleu with source.
        src_max_idx = select_bleu(srcs[i], cor_preds)
        src_best_bleu.append(cor_preds[src_max_idx])
        
        sim = util.pytorch_cos_sim(srcs_embeds[i], cor_embeds)
        sbert_idx = sim.argmax().item()
        sbert_best_bleu.append(cor_preds[sbert_idx])
        
        #all_sim.append(sim.tolist()[0])
        for j, v in enumerate(sim.tolist()[0]):
            if v >= args.thres:
                remain_pred.append(cor_preds[j])
                remain_sim.append(v)
            all_sim.append(v)
            
        remain_num.append(len(remain_pred))
        remain.append(remain_pred)
        
        #print(remain_pred)
        # exit()
        
        if len(remain_pred) != 0:
            #print(srcs[i])
            b1_l, b2_l, b3_l, b4_l = compute_bleu_1vn(srcs[i], remain_pred)
            
            #print(b4_l)
            #exit()
            
            all_b1.extend(b1_l)
            all_b2.extend(b2_l)
            all_b3.extend(b3_l)
            all_b4.extend(b4_l)
        
        
    print("Remain Rate: ", sum(remain_num)/len(preds))
    print("Remain Sem Similarity: ", sum(remain_sim)/len(remain_sim))
    print("All Sem Similarity: ", sum(all_sim)/len(all_sim))

    print("Self-BLEU: ", compute_self_bleu(remain))
    
    # semantic quality. BLEU
    refs_bleu = [[r.split()] for r in refs]
    srcs_bleu = [[s.split()] for s in srcs]
    
    ref_best_bleu = [s.split() for s in ref_best_bleu]
    src_best_bleu = [s.split() for s in src_best_bleu]
    sbert_best_bleu = [s.split() for s in sbert_best_bleu]

    print(refs_bleu[0])
    print(srcs_bleu[0])
    print(ref_best_bleu[0])
    print(src_best_bleu[0])
    print(sbert_best_bleu[0])
    
    
    # select best bleu with reference.
    ref_bleu1, ref_bleu2, ref_bleu3, ref_bleu4 = compute_bleu(refs_bleu, ref_best_bleu)
    print("select best bleu with reference. ")
    print("bleu-1: {:.4f}, bleu-2: {:.4f}, bleu-3: {:.4f}, bleu-4: {:.4f}".format(ref_bleu1, ref_bleu2, ref_bleu3, ref_bleu4))
    print()
    
    # select best bleu with source.
    src_bleu1, src_bleu2, src_bleu3, src_bleu4 = compute_bleu(refs_bleu, src_best_bleu)
    print("select best bleu with source. best vs reference.")
    print("bleu-1: {:.4f}, bleu-2: {:.4f}, bleu-3: {:.4f}, bleu-4: {:.4f}".format(src_bleu1, src_bleu2, src_bleu3, src_bleu4))   
    print()
    
    # select best sentence-bert with source.
    sbert_bleu1, sbert_bleu2, sbert_bleu3, sbert_bleu4 = compute_bleu(refs_bleu, sbert_best_bleu)
    print("select best sentence-bert with source. best vs reference.")
    print("bleu-1: {:.4f}, bleu-2: {:.4f}, bleu-3: {:.4f}, bleu-4: {:.4f}".format(sbert_bleu1, sbert_bleu2, sbert_bleu3, sbert_bleu4))  
    print()
    
    
    # select best sentence-bert with source.
    src_best_bleu1, src_best_bleu2, src_best_bleu3, src_best_bleu4 = compute_bleu(srcs_bleu, sbert_best_bleu)
    print("select best sentence-bert with source. best vs source")
    print("bleu-1: {:.4f}, bleu-2: {:.4f}, bleu-3: {:.4f}, bleu-4: {:.4f}".format(src_best_bleu1, src_best_bleu2, src_best_bleu3, src_best_bleu4))  
    print()
    
    print("Statistic BLEU Score against Source Sentence.")
    
    print("Bleu-1 score")
    pprint_min_max_ave(all_b1)
    print("Bleu-2 score")
    pprint_min_max_ave(all_b2)
    
    print("Bleu-3 score")
    pprint_min_max_ave(all_b3)
    print("Bleu-4 score")
    pprint_min_max_ave(all_b4)

    
    
    #multi_save(srcs, remain, args.out)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="To obtain the sentences embedding by BERT.")
    parser.add_argument('--src', type=str)
    parser.add_argument('--pred', type=str)
    parser.add_argument('--ref', type=str)
    parser.add_argument('--thres', type=float, default=0.7)
    parser.add_argument('--out', type=str)
    args = parser.parse_args()
    main(args)