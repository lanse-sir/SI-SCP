import os
import torch
import torch.nn as nn
from transformer_layer import PositionalEncoding, PositionalEmbedding, EncoderLayer
from embeddings import Embeddings


class SquenceEncoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, d_word_vec, n_layers, n_head, d_k, d_v, d_model, d_inner, word_embeddings,
            dropout=0.1, n_position=512,
            scale_emb=True, pe=True):

        super(SquenceEncoder, self).__init__()

        self.src_word_emb = word_embeddings
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

    def forward(self, src_word_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.src_word_emb(src_word_seq)

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
            self, n_layers, n_head, d_k, d_v, d_model, d_inner, embeddings,
            dropout=0.1, scale_emb=True):

        super(TreeEncoder, self).__init__()

        self.src_word_emb = embeddings
        self.level_emb = nn.Embedding(10, d_model, padding_idx=0)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, height_tensor, tree_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.src_word_emb(src_seq)

        if self.scale_emb:
            enc_output *= self.d_model ** 0.5

        # add level embedding.
        enc_output = self.dropout(enc_output + self.level_emb(height_tensor))

        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=tree_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output


class transformer_encoders(nn.Module):
    def __init__(self, args, word_vocab, parse_vocab, word_embedding=None):
        super(transformer_encoders, self).__init__()
        self.args = args
        self.word_vocab = word_vocab
        self.parse_vocab = parse_vocab
        self.emb_size = args.embed_size
        self.max_seq_len = args.max_seq_len

        # self.sos_id = word_vocab['<s>']
        # self.eos_id = word_vocab['</s>']
        # self.pad_id = word_vocab['<PAD>']
        # self.unk_id = word_vocab['<unk>']

        self.word_emb = Embeddings(len(self.word_vocab), args.embed_size, add_position_embedding=False,
                                   padding_idx=0)
        if word_embedding is not None:
            self.word_emb.weight.data.copy_(torch.from_numpy(word_embedding))
        self.parse_emb = Embeddings(len(self.parse_vocab), args.embed_size, args.dropout,
                                    add_position_embedding=False, padding_idx=0)

        self.sent_encoder = SquenceEncoder(
            d_word_vec=args.embed_size,
            n_layers=args.sent_enc_layers,
            n_head=args.head,
            d_k=args.dk,
            d_v=args.dv,
            d_model=args.d_model,
            d_inner=args.d_inner_hid,
            word_embeddings=self.word_emb,
            # parse_embeddings=self.parse_emb,
            scale_emb=True,
            pe=True,
            n_position=args.max_seq_len)

        self.pos_encoder = SquenceEncoder(
            d_word_vec=args.embed_size,
            n_layers=args.sent_enc_layers,
            n_head=args.head,
            d_k=args.dk,
            d_v=args.dv,
            d_model=args.d_model,
            d_inner=args.d_inner_hid,
            word_embeddings=self.parse_emb,
            scale_emb=True,
            pe=True,
            n_position=args.max_seq_len)

        if self.args.tree:
            print("Init Tree Transformer ......")
            self.parse_encoder = TreeEncoder(
                n_layers=args.parse_enc_layers,
                n_head=args.head,
                d_k=args.dk,
                d_v=args.dv,
                d_model=args.d_model,
                d_inner=args.d_inner_hid,
                embeddings=self.parse_emb,
                scale_emb=True)
        else:
            print("Using Sequence Parse Transformer ......")
            self.parse_encoder = SquenceEncoder(
                d_word_vec=args.embed_size,
                n_layers=args.sent_enc_layers,
                n_head=args.head,
                d_k=args.dk,
                d_v=args.dv,
                d_model=args.d_model,
                d_inner=args.d_inner_hid,
                word_embeddings=self.parse_emb,
                scale_emb=True,
                pe=True,
                n_position=args.max_seq_len)

        self.proj = nn.Linear(args.d_model * 2, args.d_model)

    def query_encode(self, sent_seq, sent_mask, parse_seq, parse_mask):
        sent_outputs = self.sent_encoder(sent_seq, sent_mask)
        pos_outputs = self.pos_encoder(parse_seq, parse_mask)
        sent_pos_output = torch.cat((sent_outputs[:, 0], pos_outputs[:, 0]), dim=1)
        sent_pos_output = self.proj(sent_pos_output)
        return sent_pos_output

    def corpus_encode(self, tree_tensor, tree_mask, tree_height=None):
        if self.args.tree:
            parse_outputs = self.parse_encoder(tree_tensor, tree_height, tree_mask)
        else:
            parse_outputs = self.parse_encoder(tree_tensor, tree_mask)
        return parse_outputs[:, 0]

    def forward(self, input_tensor):
        src_word_seq = input_tensor["src_seq"]
        src_mask = input_tensor["src_mask"].unsqueeze(-2)
        sent_outputs = self.sent_encoder(src_word_seq, src_mask)

        src_pos_seq = input_tensor["src_pos_seq"]
        src_pos_mask = input_tensor["src_pos_mask"].unsqueeze(-2)
        pos_outputs = self.pos_encoder(src_pos_seq, src_pos_mask)

        if self.args.tree:
            tree_tensor = input_tensor["tree_tensor"]
            tree_height = input_tensor["tree_height"]
            tree_mask = input_tensor["tree_mask"]
            parse_outputs = self.parse_encoder(tree_tensor, tree_height, tree_mask)
        else:
            tree_tensor = input_tensor["tree_tensor"]
            tree_mask = input_tensor["tree_mask"].unsqueeze(-2)
            parse_outputs = self.parse_encoder(tree_tensor, tree_mask)

        sent_pos_output = torch.cat((sent_outputs[:, 0], pos_outputs[:, 0]), dim=1)
        sent_pos_output = self.proj(sent_pos_output)
        # Add non-linearized transformation.
        # sent_pos_output = torch.tanh(sent_pos_output)

        parse_output = parse_outputs[:, 0]
        return sent_pos_output, parse_output

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
