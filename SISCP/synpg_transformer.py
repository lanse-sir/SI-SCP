import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from autocg.Modules.transformer_layer import EncoderLayer, PositionalEncoding, PositionalEmbedding, MultiHeadAttention, \
    PositionwiseFeedForward
from autocg.networks.embeddings import Embeddings


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        # two cross attentions.
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.parse_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)

        self.gate = nn.Linear(d_model * 2, d_model)
        # self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        # self.fusion = nn.Linear(d_model * 2, d_model, bias=False)
        self.w_1 = nn.Linear(d_model, d_inner)  # position-wise
        self.w_2 = nn.Linear(d_inner, d_model)  # position-wise
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self, dec_input, enc_output, enc_parse_output,
            slf_attn_mask=None, dec_enc_attn_mask=None, dec_parse_enc_mask=None):
        dec_slf_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)

        dec_parse_output, dec_parse_attn = self.parse_attn(dec_slf_output, enc_parse_output, enc_parse_output,
                                                           mask=dec_parse_enc_mask)

        dec_output, dec_enc_attn = self.enc_attn(
            dec_parse_output, enc_output, enc_output, mask=dec_enc_attn_mask)

        # dec_sent_output, dec_enc_attn = self.enc_attn(
        # dec_slf_output, enc_output, enc_output, mask=dec_enc_attn_mask)

        # gate mechnism.
        # gate_value = torch.sigmoid(self.gate(torch.cat([dec_sent_output, dec_parse_output], dim=2)))
        # dec_output = gate_value * dec_sent_output + (1-gate_value)*dec_parse_output

        residual = dec_output

        dec_output = self.w_2(F.relu(self.w_1(dec_output)))
        dec_output = self.dropout(dec_output)
        dec_output += residual

        dec_output = self.layer_norm(dec_output)

        return dec_output, dec_slf_attn, dec_parse_attn, dec_enc_attn


class Tree_EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(Tree_EncoderLayer, self).__init__()
        self.td_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.lr_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)

        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, lr_attn_mask=None, td_attn_mask=None):
        enc_output, enc_slf_attn = self.td_attn(
            enc_input, enc_input, enc_input, mask=td_attn_mask)
        
    
        enc_output, enc_slf_attn = self.lr_attn(
            enc_output, enc_output, enc_output, mask=lr_attn_mask)

        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, d_word_vec, n_layers, n_head, d_k, d_v, d_model, d_inner, embeddings, dropout=0.1, n_position=500,
            scale_emb=True, pe=True):

        super(Encoder, self).__init__()

        self.src_word_emb = embeddings
        if pe:
            self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        else:
            self.position_enc = PositionalEmbedding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        if self.src_word_emb is not None:  # don't need embedding layer.
            enc_output = self.src_word_emb(src_seq)
        else:
            enc_output = src_seq

        if self.scale_emb:
            enc_output *= self.d_model ** 0.5
        enc_output = self.dropout(self.position_enc(enc_output))

        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output


class TreeEncoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, d_word_vec, n_layers, n_head, d_k, d_v, d_model, d_inner, embeddings, dropout=0.1, n_levels=50,
            scale_emb=True, pe=False):

        super(TreeEncoder, self).__init__()

        self.src_word_emb = embeddings

        if pe:
            self.position_enc = PositionalEncoding(d_word_vec, n_position=n_levels)
        else:
            self.level_embedding = Embeddings(n_levels, d_model, add_position_embedding=False, padding_idx=0)
            # self.position_enc = PositionalEmbedding(d_word_vec, n_position=n_position)

        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, level_seq, mask, return_attns=False):

        enc_slf_attn_list = []
        # src_mask = src_mask.view(-1, height).unsqueeze(-2)

        # from top to down.
        # src_mask = get_pad_mask(src_seq, 0) & get_subsequent_mask(src_seq)

        # -- Forward
        enc_output = self.src_word_emb(src_seq)
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5

        level_emb = self.level_embedding(level_seq)
        enc_output = self.dropout(enc_output + level_emb)

        # enc_output = self.dropout(self.position_enc(enc_output))

        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        # get the deepest node representation.
        # enc_output = torch.gather(enc_output, 1, (tree_height - 1))
        # enc_output = enc_output.squeeze(1).view(batch_size, paths, -1)
        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output


class Sibling_TreeEncoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, d_word_vec, n_layers, n_head, d_k, d_v, d_model, d_inner, embeddings, dropout=0.1, n_levels=50,
            scale_emb=True, pe=False, hie_type="cat"):

        super(Sibling_TreeEncoder, self).__init__()

        self.src_word_emb = embeddings
        self.hie_type = hie_type

        if self.hie_type == "cat":
            self.posi_emb_dim = d_model // 2
        else:
            self.posi_emb_dim = d_model

        print("Merging Type: ", self.hie_type)
        print("Hierarchical Embedding dim: ", self.posi_emb_dim)

        if pe:
            self.position_enc = PositionalEncoding(d_word_vec, n_position=n_levels)
        else:
            self.level_embedding = Embeddings(n_levels, self.posi_emb_dim, add_position_embedding=False, padding_idx=0)
            self.sibling_embedding = Embeddings(n_levels, self.posi_emb_dim, add_position_embedding=False,
                                                padding_idx=0)

        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            Tree_EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, sibling_posi_seq, level_seq, lr_mask, td_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.src_word_emb(src_seq)
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5

        level_emb = self.level_embedding(level_seq)
        sibling_emb = self.sibling_embedding(sibling_posi_seq)

        if self.hie_type == "cat":
            hie_emb = torch.cat([level_emb, sibling_emb], dim=-1)
        else:
            hie_emb = level_emb + sibling_emb
            
            #hie_emb = level_emb
            
            
        enc_output = self.dropout(enc_output + hie_emb)

        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, lr_attn_mask=lr_mask, td_attn_mask=td_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        # get the deepest node representation.
        # enc_output = torch.gather(enc_output, 1, (tree_height - 1))
        # enc_output = enc_output.squeeze(1).view(batch_size, paths, -1)
        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, embeddings, n_position=500, dropout=0.1, scale_emb=False, pe=True):

        super(Decoder, self).__init__()

        self.trg_word_emb = embeddings
        if pe:
            self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        else:
            self.position_enc = PositionalEmbedding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, enc_parse_output, enc_parse_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list, dec_parse_attn_list = [], [], []

        # -- Forward
        dec_output = self.trg_word_emb(trg_seq)
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_parse_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, enc_parse_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask,
                dec_parse_enc_mask=enc_parse_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []
            dec_parse_attn_list += [dec_parse_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list, dec_parse_attn_list
        return dec_output


class synpg_transformer(nn.Module):
    def __init__(self, args, word_vocab, parse_vocab, word_embedding=None):
        super(synpg_transformer, self).__init__()
        self.args = args
        # sibling relation.
        self.sibling = args.sibling

        self.word_vocab = word_vocab
        self.parse_vocab = parse_vocab
        self.vocab_size = len(self.word_vocab)
        self.emb_size = args.embed_size
        self.max_seq_len = args.max_seq_len

        self.id2word = {idx: word for word, idx in self.word_vocab.items()}
        self.sos_id = word_vocab['<s>']
        self.eos_id = word_vocab['</s>']
        self.pad_id = word_vocab['<PAD>']
        self.unk_id = word_vocab['<unk>']

        self.word_emb = Embeddings(len(self.word_vocab), args.embed_size, add_position_embedding=False,
                                   padding_idx=0)
        if word_embedding is not None:
            self.word_emb.weight.data.copy_(torch.from_numpy(word_embedding))
        self.parse_emb = Embeddings(len(self.parse_vocab), args.tree_embed_size, args.dropout,
                                    add_position_embedding=False, padding_idx=0)

        self.sent_encoder = Encoder(
            d_word_vec=args.embed_size,
            n_layers=args.enc_num_layers,
            n_head=args.head,
            d_k=args.dk,
            d_v=args.dv,
            d_model=args.d_model,
            d_inner=args.d_inner_hid,
            embeddings=self.word_emb,
            scale_emb=args.scale,
            pe=args.pe,
            n_position=args.max_seq_len)

        if self.sibling:
            self.parse_encoder = Sibling_TreeEncoder(d_word_vec=args.embed_size,
                                                     n_layers=args.parse_td_enc_layers,
                                                     n_head=args.head,
                                                     d_k=args.dk,
                                                     d_v=args.dv,
                                                     d_model=args.d_model,
                                                     d_inner=args.d_inner_hid,
                                                     embeddings=self.parse_emb,
                                                     scale_emb=args.scale,
                                                     pe=args.parse_pe,
                                                     n_levels=50,
                                                     hie_type=args.hie_type)

        else:
            self.parse_encoder_td = TreeEncoder(
                d_word_vec=args.embed_size,
                n_layers=args.parse_td_enc_layers,
                n_head=args.head,
                d_k=args.dk,
                d_v=args.dv,
                d_model=args.d_model,
                d_inner=args.d_inner_hid,
                embeddings=self.parse_emb,
                scale_emb=args.scale,
                pe=args.parse_pe,
                n_levels=50)
        
        if self.args.parse_lr_enc_layers > 0:
            self.parse_encoder_lr = Encoder(
                d_word_vec=args.d_model,
                n_layers=args.parse_lr_enc_layers,
                n_head=args.head,
                d_k=args.dk,
                d_v=args.dv,
                d_model=args.d_model,
                d_inner=args.d_inner_hid,
                embeddings=None,  # don't need the embedding layer.
                scale_emb=args.scale,
                pe=args.pe,
                n_position=args.max_seq_len)
        else:
            print("No Sequence Parse Encoder...")
        
        self.decoder = Decoder(
            d_word_vec=args.embed_size,
            n_layers=args.dec_num_layers,
            n_head=args.head,
            d_k=args.dk,
            d_v=args.dv,
            d_model=args.d_model,
            d_inner=args.d_inner_hid,
            embeddings=self.word_emb,
            scale_emb=args.scale,
            pe=args.pe,
            n_position=args.max_seq_len)

        # self.max_pool = nn.MaxPool2d((self.args.pool_size, 1), stride=(self.args.pool_size, 1))
        self.tgt_project = nn.Linear(args.d_model, len(self.word_vocab), bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print(
            "Sentence enc layer: {}, Parse tree enc layer: {}, Parse Seq enc layer: {}, Sentence Dec layer: {}, Positional Encoding: {}".format(
                args.enc_num_layers,
                args.parse_td_enc_layers,
                args.parse_lr_enc_layers,
                args.dec_num_layers,
                args.pe))

    def to_variable(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def add_noise(self, variable, pad_index: int, drop_probability: float = 0.1,
                  shuffle_max_distance: int = 3) -> Variable:
        def perm(i):
            return i[0] + (shuffle_max_distance + 1) * np.random.random()

        new_variable = np.zeros((variable.size(0), variable.size(1)), dtype='int')
        variable = variable.data.cpu().numpy()
        for b in range(variable.shape[0]):
            sequence = variable[b]
            sequence = sequence[sequence != pad_index]
            sequence, reminder = sequence[:-1], sequence[-1:]
            if len(sequence) > 2:
                sequence = sequence[np.random.random_sample(len(sequence)) > drop_probability]
                sequence = [x for _, x in sorted(enumerate(sequence), key=perm)]
            sequence = np.concatenate((sequence, reminder), axis=0)
            sequence = list(np.pad(sequence, (0, variable.shape[1] - len(sequence)), 'constant',
                                   constant_values=pad_index))
            new_variable[b, :] = sequence
        return torch.LongTensor(new_variable).cuda()

    def sentence_encoding(self, input_tensor):
        src_sent_tensor = input_tensor['src_seq']

        # multi-head attention need 4 dim.
        src_mask = input_tensor['src_mask'].unsqueeze(-2)
        sent_encoder_outputs = self.sent_encoder(src_sent_tensor, src_mask)
        return sent_encoder_outputs

    def parse_encoding(self, input_tensor):
        tree_seq_tensor = input_tensor['tree_tensor']
        td_tree_mask = input_tensor['tree_mask']
        level_seqs_tensor = input_tensor['tree_height']

        lr_mask = input_tensor['sibling_mask']
        lr_posi = input_tensor['sibling_posi']

        leave_node_idx = input_tensor['leave_node']
        leave_node_mask = input_tensor['leave_mask'].unsqueeze(-2)

        if self.sibling:
            parse_encoder_outputs = self.parse_encoder(tree_seq_tensor, lr_posi, level_seqs_tensor, lr_mask,
                                                       td_tree_mask)
            # parse_encoder_outputs = torch.gather(parse_encoder_outputs, 1,
            # leave_node_idx.unsqueeze(-1).repeat(1, 1,
            # parse_encoder_outputs.size(2)))
        else:
            parse_encoder_outputs = self.parse_encoder_td(tree_seq_tensor, level_seqs_tensor, td_tree_mask)

        # gather leaf node.
        parse_encoder_outputs = torch.gather(parse_encoder_outputs, 1,
                                             leave_node_idx.unsqueeze(-1).repeat(1, 1,
                                                                                 parse_encoder_outputs.size(2)))
        # add left-to-right layers.
        if self.args.parse_lr_enc_layers > 0: 
            parse_encoder_outputs = self.parse_encoder_lr(parse_encoder_outputs, leave_node_mask)

        return parse_encoder_outputs

    def forward(self, input_tensor):
        src_mask = input_tensor['src_mask'].unsqueeze(-2)

        # tree_seq_tensor = input_tensor['tree_tensor']
        # tree_mask = input_tensor['tree_mask']
        # level_seqs_tensor = input_tensor['tree_height']
        #
        # leave_node_idx = input_tensor['leave_node']
        leave_node_mask = input_tensor['leave_mask'].unsqueeze(-2)

        inp_sent_tensor = input_tensor['inp_seq']

        # add noises ...
        # if self.args.noise:
        #     src_sent_tensor = self.add_noise(src_sent_tensor, self.pad_id, self.args.word_drop, self.args.shuffle)
        #     src_mask = get_pad_mask(src_sent_tensor, self.pad_id)

        sent_encoder_outputs = self.sentence_encoding(input_tensor)

        parse_encoder_outputs = self.parse_encoding(input_tensor)
        # next is the decoder section .
        # Decode
        trg_mask = get_pad_mask(inp_sent_tensor, self.pad_id) & get_subsequent_mask(inp_sent_tensor)
        dec_output, dec_slf_attns, dec_sent_attns, dec_parse_attns = self.decoder(inp_sent_tensor, trg_mask,
                                                                                  sent_encoder_outputs,
                                                                                  src_mask, parse_encoder_outputs,
                                                                                  leave_node_mask, return_attns=True)
        seq_logit = self.tgt_project(dec_output)

        # compute loss
        # decode loss.
        # if tgt_sent_tensor.size(0) == batch_size:
        #     output_ids = tgt_sent_tensor.contiguous().transpose(1, 0)
        # output_ids = output_ids[1:].contiguous().view(-1)
        # tgt_sent_log_scores = torch.gather(logprods.view(-1, logprods.size(2)), 1, output_ids.unsqueeze(1)).squeeze(1)
        # tgt_sent_log_scores = tgt_sent_log_scores * (1.0 - torch.eq(output_ids, self.pad_id).float())

        # batch size .
        # sent_scores = tgt_sent_log_scores.view(-1, batch_size).sum(dim=0)
        # s_reconstruct_loss = -torch.sum(sent_scores) / batch_size
        return seq_logit.view(-1, seq_logit.size(2)), dec_parse_attns

    def save(self, path):
        dir_name = os.path.dirname(path)
        # remove file name, return directory.
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        params = {
            'args': self.args,
            'word_vocab': self.word_vocab,
            'parse_vocab': self.parse_vocab,
            'state_dict': self.state_dict()
        }
        torch.save(params, path)
