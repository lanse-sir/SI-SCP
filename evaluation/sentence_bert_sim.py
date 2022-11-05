from sentence_transformers import SentenceTransformer, util
import numpy as np
import argparse


model_path = '../eval_tools/distilroberta-base-paraphrase-v1'
model = SentenceTransformer(model_path)


def load_sentence(file_name):
    sentences = []
    with open(file_name, 'r', encoding='utf-8') as f_r:
        for line in f_r:
            sentences.append(line.strip())
    return sentences

def main(args):
    corpus = load_sentence(args.input_file)
    refs = load_sentence(args.ref_file)
    corpus_embeds = model.encode(corpus, batch_size=64)
    refs_embeds = model.encode(refs, batch_size=64)
    print(corpus_embeds.shape)
    print(refs_embeds.shape)
    
    all_sim = []
    for i in range(len(refs)):
        sim = util.pytorch_cos_sim(corpus_embeds[i], refs_embeds[i])
        all_sim.append(sim[0][0].item())
    average_sim = sum(all_sim) / len(all_sim)
    print('Average Similarity:', average_sim)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="To obtain the sentences embedding by BERT.")
    parser.add_argument('--input_file',type=str)
    parser.add_argument('--ref_file',type=str)
    args = parser.parse_args()
    main(args)