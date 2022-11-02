''' This module will handle the text generation with beam search. '''

import torch
import torch.nn as nn
import torch.nn.functional as F
from autocg.utils.nn_funcs import id2word
from torch.autograd import Variable
from syn_control_pg.synpg_transformer import get_subsequent_mask, get_pad_mask


# from transformer.Models import Transformer, get_pad_mask, get_subsequent_mask


class Translator(nn.Module):
    ''' Load a trained model and translate in beam search fashion. '''

    def __init__(
            self, model, beam_size, max_seq_len):

        super(Translator, self).__init__()

        self.alpha = 0.7
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.src_pad_idx = model.pad_id
        self.trg_pad_idx = self.src_pad_idx
        self.trg_bos_idx = model.sos_id
        self.trg_eos_idx = model.eos_id

        self.model = model
        self.model.eval()

        self.register_buffer('init_seq', torch.LongTensor([[self.trg_bos_idx]]).cuda())
        self.register_buffer(
            'blank_seqs',
            torch.full((beam_size, max_seq_len), self.trg_pad_idx, dtype=torch.long).cuda())
        self.blank_seqs[:, 0] = self.trg_bos_idx
        self.register_buffer(
            'len_map',
            torch.arange(1, max_seq_len + 1, dtype=torch.long).cuda().unsqueeze(0))

    def _model_decode(self, trg_seq, enc_output, src_mask, parse_encoder_output, tree_path_mask, return_attns=False):
        trg_mask = get_subsequent_mask(trg_seq)
        if return_attns:
            dec_output, _, _, syn_attn = self.model.decoder(trg_seq, trg_mask, enc_output, src_mask,
                                                            parse_encoder_output,
                                                            tree_path_mask,
                                                            return_attns=return_attns)
            return F.softmax(self.model.tgt_project(dec_output), dim=-1), syn_attn

        else:
            dec_output = self.model.decoder(trg_seq, trg_mask, enc_output, src_mask, parse_encoder_output,
                                            tree_path_mask,
                                            return_attns=return_attns)
            return F.softmax(self.model.tgt_project(dec_output), dim=-1), 0.0

    def _get_init_state(self, src_seq, src_mask):
        beam_size = self.beam_size

        enc_output = self.model.sent_encoder(src_seq, src_mask)
        dec_output = self._model_decode(self.init_seq, enc_output, src_mask)

        best_k_probs, best_k_idx = dec_output[:, -1, :].topk(beam_size)

        scores = torch.log(best_k_probs).view(beam_size)
        gen_seq = self.blank_seqs.clone().detach()
        gen_seq[:, 1] = best_k_idx[0]
        enc_output = enc_output.repeat(beam_size, 1, 1)
        return enc_output, gen_seq, scores

    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step):
        assert len(scores.size()) == 1

        beam_size = self.beam_size

        # Get k candidates for each beam, k^2 candidates in total.
        best_k2_probs, best_k2_idx = dec_output[:, -1, :].topk(beam_size)

        # Include the previous scores.
        scores = torch.log(best_k2_probs).view(beam_size, -1) + scores.view(beam_size, 1)

        # Get the best k candidates from k^2 candidates.
        scores, best_k_idx_in_k2 = scores.view(-1).topk(beam_size)

        # Get the corresponding positions of the best k candidiates.
        best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size
        best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]

        # Copy the corresponding previous tokens.
        gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step]
        # Set the best tokens in this beam search step
        gen_seq[:, step] = best_k_idx

        return gen_seq, scores

    def translate_sentence(self, input_tensor):
        # Only accept batch size equals to 1 in this function.
        # TODO: expand to batch operation.
        src_seq = input_tensor['src_seq']

        assert src_seq.size(0) == 1

        src_pad_idx, trg_eos_idx = self.src_pad_idx, self.trg_eos_idx
        max_seq_len, beam_size, alpha = self.max_seq_len, self.beam_size, self.alpha

        with torch.no_grad():
            src_mask = self.model.get_pad_mask(src_seq, src_pad_idx)
            enc_output, gen_seq, scores = self._get_init_state(src_seq, src_mask)

            ans_idx = 0  # default
            for step in range(2, max_seq_len):  # decode up to max length
                dec_output = self._model_decode(gen_seq[:, :step], enc_output, src_mask)
                gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, dec_output, scores, step)

                # Check if all path finished
                # -- locate the eos in the generated sequences
                eos_locs = gen_seq == trg_eos_idx
                # -- replace the eos with its position for the length penalty use
                seq_lens, _ = self.len_map.masked_fill(~eos_locs, max_seq_len).min(1)
                # -- check if all beams contain eos
                if (eos_locs.sum(1) > 0).sum(0).item() == beam_size:
                    # TODO: Try different terminate conditions.
                    _, ans_idx = scores.div(seq_lens.float() ** alpha).max(0)
                    ans_idx = ans_idx.item()
                    break
        return [id2word(gen_seq[ans_idx][:seq_lens[ans_idx]].tolist(), self.model.parse_vocab)]

    def translate_sentences(self, input_tensor):
        src_var = input_tensor['src_seq']
        src_mask = input_tensor['src_mask'].unsqueeze(-2)
        batch_size = src_var.size(0)
        beam_size = self.beam_size
        # tree_seq_tensor = input_tensor['tree_tensor']
        # tree_mask = input_tensor['tree_mask']
        # level_seqs_tensor = input_tensor['tree_height']
        #
        # leave_node_idx = input_tensor['leave_node']
        leave_node_mask = input_tensor['leave_mask'].unsqueeze(-2)

        enc_outputs = self.model.sentence_encoding(input_tensor)
        parse_encoder_outputs = self.model.parse_encoding(input_tensor)

        # parse_encoder_outputs = self.model.parse_encoder_td(tree_seq_tensor, level_seqs_tensor, tree_mask)
        #
        # parse_encoder_outputs = torch.gather(parse_encoder_outputs, 1,
        #                                      leave_node_idx.unsqueeze(-1).repeat(1, 1, parse_encoder_outputs.size(2)))
        # # add left-to-right layer.
        # parse_encoder_outputs = self.model.parse_encoder_lr(parse_encoder_outputs, leave_node_mask)

        # start to decode .
        translations = [[] for i in range(batch_size)]
        pending = set(range(batch_size))

        # hidden = None
        # context_lengths *= beam_size
        # context_mask = self.mask(context_lengths)
        enc_outputs = enc_outputs.repeat(beam_size, 1, 1)
        src_mask = src_mask.repeat(beam_size, 1, 1)

        parse_encoder_outputs = parse_encoder_outputs.repeat(beam_size, 1, 1)
        leave_node_mask = leave_node_mask.repeat(beam_size, 1, 1)

        prev_words = torch.full((beam_size * batch_size, self.max_seq_len), self.trg_pad_idx, dtype=torch.long).cuda()
        prev_words[:, 0] = self.trg_bos_idx

        translation_scores = batch_size * [-float('inf')]
        hypotheses = batch_size * [(0.0, [])] + (beam_size - 1) * batch_size * [
            (-float('inf'), [])]  # (score, translation)

        # while len(pending) > 0:
        for step in range(1, self.max_seq_len):
            if len(pending) <= 0:
                break
            # Each iteration should update: prev_words, hidden, output
            input_var = prev_words[:, :step]
            # logprobs, hidden, output = self.decoder(var, ones, self.decoder_embeddings, hidden, context, context_mask,
            #                                         output, self.generator)
            softmax_outout, _ = self._model_decode(input_var, enc_outputs, src_mask, parse_encoder_outputs,
                                                   leave_node_mask)
            log_softmax_output = torch.log(softmax_outout)[:, -1, :]  # get the final tensor.

            # prev_words = log_softmax_output.max(dim=2)[1].squeeze().data.cpu().numpy().tolist()

            word_scores, words = log_softmax_output.topk(k=beam_size + 1, dim=1, sorted=False)
            word_scores = word_scores.data.cpu().numpy().tolist()  # (beam_size*batch_size) * (beam_size+1)
            words = words.data.cpu().numpy().tolist()

            for sentence_index in pending.copy():
                candidates = []  # (score, index, word)
                for rank in range(beam_size):
                    index = sentence_index + rank * batch_size
                    for i in range(beam_size + 1):
                        word = words[index][i]
                        score = hypotheses[index][0] + word_scores[index][i]
                        if word != self.trg_eos_idx:
                            candidates.append((score, index, word))
                        elif score > translation_scores[sentence_index]:
                            translations[sentence_index] = hypotheses[index][1] + [word]
                            translation_scores[sentence_index] = score
                best = []  # score, word, translation, hidden, output
                for score, current_index, word in sorted(candidates, reverse=True)[:beam_size]:
                    translation = hypotheses[current_index][1] + [word]
                    best.append(
                        (score, word, translation))
                for rank, (score, word, translation) in enumerate(best):
                    next_index = sentence_index + rank * batch_size
                    hypotheses[next_index] = (score, translation)
                    prev_words[next_index, step] = word
                if len(hypotheses[sentence_index][1]) >= self.max_seq_len or \
                        translation_scores[sentence_index] > hypotheses[sentence_index][0]:
                    pending.discard(sentence_index)
                    if len(translations[sentence_index]) == 0:
                        translations[sentence_index] = hypotheses[sentence_index][1]
                        translation_scores[sentence_index] = hypotheses[sentence_index][0]
        res = id2word(translations, self.model.word_vocab)
        # res = [' '.join(s).replace('@@ ', '') for s in res]
        return res, translation_scores

    def greedy_decode(self, input_tensor, return_attns=False):
        src_var = input_tensor['src_seq']
        src_mask = input_tensor['src_mask'].unsqueeze(-2)
        batch_size = src_var.size(0)
        # tree_seq_tensor = input_tensor['tree_tensor']
        # tree_mask = input_tensor['tree_mask']
        # level_seqs_tensor = input_tensor['tree_height']
        #
        # leave_node_idx = input_tensor['leave_node']
        leave_node_mask = input_tensor['leave_mask'].unsqueeze(-2)

        enc_outputs = self.model.sentence_encoding(input_tensor)
        parse_encoder_outputs = self.model.parse_encoding(input_tensor)

        # enc_outputs = self.model.sent_encoder(src_var, src_mask)

        # tree encoder.
        # parse_encoder_outputs = self.model.parse_encoder_td(tree_seq_tensor, level_seqs_tensor, tree_mask)
        #
        # parse_encoder_outputs = torch.gather(parse_encoder_outputs, 1,
        #                                      leave_node_idx.unsqueeze(-1).repeat(1, 1, parse_encoder_outputs.size(2)))
        # # add left-to-right layers.
        # parse_encoder_outputs = self.model.parse_encoder_lr(parse_encoder_outputs, leave_node_mask)

        ### start to decode .
        translations = [[] for i in range(batch_size)]
        attentions = [[] for i in range(batch_size)]

        pending = set(range(batch_size))

        prev_words = torch.full((batch_size, self.max_seq_len), self.trg_pad_idx, dtype=torch.long).cuda()

        prev_words[:, 0] = self.trg_bos_idx

        for step in range(1, self.max_seq_len):
            if len(pending) <= 0:
                break
            # Each iteration should update: prev_words, hidden, output
            input_var = prev_words[:, :step]
            # logprobs, hidden, output = self.decoder(var, ones, self.decoder_embeddings, hidden, context, context_mask,
            #                                         output, self.generator)

            softmax_outout, syn_attn = self._model_decode(input_var, enc_outputs, src_mask, parse_encoder_outputs,
                                                          leave_node_mask, return_attns=return_attns)

            # print(softmax_outout)
            # print(softmax_outout.size())
            # print(syn_attn[0].size())

            log_softmax_output = torch.log(softmax_outout)[:, -1, :]  # get the final tensor.
            prev_word = log_softmax_output.max(dim=1)[1].squeeze().data.cpu().numpy().tolist()

            # word_scores, words = log_softmax_output.topk(k=1, dim=1, sorted=False)
            # word_scores = word_scores.data.cpu().numpy().tolist()  # (beam_size*batch_size) * (beam_size+1)
            # words = words.data.cpu().numpy().tolist()

            for i in pending.copy():
                if prev_word[i] == self.trg_eos_idx:
                    pending.discard(i)
                else:
                    translations[i].append(prev_word[i])
                    attentions[i].append(syn_attn[-1][i, :, -1, :].unsqueeze(-2))

                    prev_words[i, step] = prev_word[i]
                    if len(translations[i]) >= self.max_seq_len:
                        pending.discard(i)
        res = self.ids2word(translations)
        # res = [' '.join(s).replace('@@ ', '') for s in res]
        return res, attentions

    def id2word(self, sent):
        res = []
        # id2word = {idx: word for word, idx in vocab.items()}

        for id in sent:
            w = self.model.id2word[id]
            if w == '<s>' or w == '<PAD>':
                pass
            elif w == '</s>':
                break
            else:
                res.append(w)
        return ' '.join(res)

    def ids2word(self, sents):
        res = []
        for sent in sents:
            res.append(self.id2word(sent))
        return res
