import torch
import numpy as np
from typing import List, Tuple
import math


def base_to_tokens(x: np.ndarray, base=4, channels=3):
    B = x.shape[0]
    f = base ** np.arange(0, channels)[::-1]
    return np.sum(x * f, axis=-1)


def token_to_base(x: np.ndarray, base=4, n_channels=3):
    factors = np.array([base**i for i in reversed(range(n_channels))])
    out = (x[..., np.newaxis] // factors) % base
    return out


def tokenize_data_set(vqvae, data, H_t=7, W_t=7, bs=16):
    # need to load in batches to avoid cuda out of memory error
    n = data.shape[0]
    assert n % bs == 0
    data_tok = np.zeros((n, H_t*W_t), dtype=int)

    with torch.no_grad():
        for i in range(n // bs):
            data_tok[i*bs:(i+1)*bs] = vqvae.quantize(data[i*bs:(i+1)*bs]).cpu().numpy().reshape(-1, H_t*W_t)  # (n, H, W, C) -> (n_train, 7, 7)

    return data_tok


class Preprocessor:
    def __init__(self, text: List[str], context_size=128):
        self.context_size = context_size

        chrs = sorted(list(set(''.join(text))))
        self.bos_idx, self.eos_idx = 0, len(chrs) + 1
        self.n_tokens = len(chrs) + 2

        self.chr_to_idx = {chr:i+1 for i, chr in enumerate(chrs)}
        self.idx_to_chr = {v:k for k,v in self.chr_to_idx.items()}
        self.idx_to_chr[0] = '<bos>'

    def preprocess(self, text: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        def passage_to_seqs(tok_passage: List[int], pad_int=0) -> np.ndarray:
            '''Splits tokenized passage into ctx length chunks. Will pad with zeros if neccessary'''
            ctx_size = self.context_size
            n_seqs = math.ceil(len(tok_passage) / ctx_size)
            return np.array([
                chunk + [pad_int] * (ctx_size - len(chunk))
                if len(chunk := tok_passage[i*ctx_size:(i+1)*ctx_size]) < ctx_size
                else chunk
                for i in range(n_seqs)
            ])

        text_tok = [
            [self.bos_idx] + [self.chr_to_idx[chr] for chr in passage] + [self.eos_idx]
            for passage in text
        ]

        X, Y = (np.zeros((0, self.context_size), dtype=int) for _ in range(2))
        for tok_passage in text_tok:
            x, y = tok_passage[:-1], tok_passage[1:]
            X, Y = np.vstack((X, passage_to_seqs(x))), np.vstack((Y, passage_to_seqs(y, pad_int=-1)))

        return X, Y

    def unpreprocess(self, data: np.ndarray) -> List[str]:
        def convert_seq(tok_seq: List[int]):
            text_seq = []
            for tok in tok_seq:
                if tok == self.eos_idx: break
                text_seq.append(self.idx_to_chr[tok])
            return ''.join(text_seq)

        return [convert_seq(seq) for seq in data.tolist()]


class PreprocessorMultiModal:
    def __init__(self, text, vqvae):
        self.vqvae = vqvae
        self.eoi, self.eot = 0, 1
        unique_words = sorted(list(set([word for s in text for word in s.split(' ')])))
        self.wtoi = {w:i+2 for i, w in enumerate(unique_words)}
        self.itow = {v:k for k, v in self.wtoi.items()}

        self.n_text_toks = len(unique_words)
        self.n_img_toks = vqvae.n_embeddings
        self.n_tokens = self.n_text_toks + self.n_img_toks + 2

        self.tok_idx_offset = 2 + len(unique_words)

    def encode_text(self, text: List[str]) -> List[List[int]]:
        return [[self.wtoi[w] for w in s.split(' ')] for s in text]


    def decode_text(self, tokens: List[List[int]]) -> List[str]:
        return [
            ' '.join([self.itow[idx] for idx in seq])
            for seq in tokens
        ]

    def combine(self, img_toks, text_toks):
        p_text_toks = [[self.eoi] + seq for seq in text_toks]
        p_img_toks = np.insert(img_toks + self.tok_idx_offset, 0, self.eot, axis=1)

        split = int(0.5 * len(img_toks))
        text_first = np.hstack([p_text_toks[:split], p_img_toks[:split]])
        img_first = np.hstack([p_img_toks[split:], p_text_toks[split:]])
        return np.vstack([text_first, img_first])


    def decode_multimodal(self, tokens: np.ndarray) -> List[Tuple[np.ndarray, str]]:
        B, _ = tokens.shape
        tok_img_len = 7*7
        text_len = 6

        eoi_mask = tokens[:, 0] == self.eoi
        eot_mask = tokens[:, 0] == self.eot

        tok_imgs = np.vstack([
            tokens[eoi_mask, 8:],
            tokens[eot_mask, 1:tok_img_len+1]
        ]).reshape(-1, 7, 7)

        text_tokens = [
            tokens[i, 1:text_len+1].tolist()
            if eoi_mask[i]
            else tokens[i, 2+tok_img_len:].tolist()
            for i in range(B)
        ]

        tok_imgs = tok_imgs - self.tok_idx_offset

        decoded_imgs = self.vqvae.decode(tok_imgs)
        decoded_text = self.decode_text(text_tokens)

        return [(decoded_imgs[i], decoded_text[i]) for i in range(B)]


def sample_multimodal(logits: torch.Tensor, preprocessor, restrict_tokens=None) -> torch.Tensor:
    assert restrict_tokens in (None, 'special', 'text', 'img')
    token_ranges = {
        'special': (0, 2),
        'text': (2, 2 + preprocessor.n_text_toks),
        'img': (2 + preprocessor.n_text_toks, logits.shape[-1])
    }

    start, end = token_ranges.get(restrict_tokens, (0, logits.shape[-1]))
    prob_dist = torch.softmax(logits[:, start:end], dim=-1)
    samples = torch.multinomial(prob_dist, num_samples=1) + start

    return samples


def generate_image(model, text_prompt: List[str], preprocessor):
    x = torch.zeros((len(text_prompt), model.max_seq_len), dtype=torch.long)
    x[:, 0] = model.n_tokens - 1  # add <bos> token
    x[:, 1] = preprocessor.eoi
    len_prompt = len(text_prompt[0].split(' '))

    encoded_text = preprocessor.encode_text(text_prompt)

    x[:, 2:len_prompt+2] = torch.tensor(preprocessor.encode_text(text_prompt))
    x[:, len_prompt+2] = preprocessor.eot

    for i in range(2 + len_prompt, model.max_seq_len):
        logits = model.forward(x)
        samples = sample_multimodal(logits[:, i, :], preprocessor, restrict_tokens='img')

        if i == model.max_seq_len - 1:
            return torch.cat((x[:, 1:], samples), dim=1)

        x[:, i+1] = samples.view(-1)


def generate_text(model, vqvae, image_prompt: np.ndarray, preprocessor):
    x = torch.zeros((len(image_prompt), model.max_seq_len), dtype=torch.long)
    x[:, 0] = model.n_tokens - 1  # add <bos> token
    x[:, 1] = preprocessor.eot
    tok_img = torch.tensor(vqvae.quantize(image_prompt).reshape(len(image_prompt), -1)) + preprocessor.tok_idx_offset
    img_tok_len = tok_img.shape[1]
    x[:, 2:img_tok_len+2] = tok_img
    x[:, img_tok_len+2] = preprocessor.eoi

    for i in range(2 + img_tok_len, model.max_seq_len):
        logits = model.forward(x)
        samples = sample_multimodal(logits[:, i, :], preprocessor, restrict_tokens='text')

        if i == model.max_seq_len - 1:
            return torch.cat((x[:, 1:], samples), dim=1)

        x[:, i+1] = samples.view(-1)


# warning: the following is complete spaghetticode.
# A better way to do this is to process the batch simultaneously.
# This can be done by cleverly masking the logits
def generate_unconditional(model, n_samples, preprocessor):
    x = torch.zeros((n_samples, model.max_seq_len), dtype=torch.long)
    x[:, 0] = model.n_tokens - 1  # add <bos> token
    for i in range(n_samples):
        text_token_count, img_token_count = 0, 0
        for j in range(model.max_seq_len):
            logits = model.forward(x[i, :][None, :])
            if j == 0:
                restrict_tokens = 'special'
            elif j == 1:
                restrict_tokens = 'img' if x[i, 1] == preprocessor.eot else 'text'

            samples = sample_multimodal(logits[:, j, :], preprocessor, restrict_tokens=restrict_tokens)

            if restrict_tokens == 'text':
                text_token_count += 1
                if text_token_count == 7:
                    samples = preprocessor.eot  # force eot token
                    restrict_tokens = 'img'
            elif restrict_tokens == 'img':
                img_token_count += 1
                if img_token_count == 50:
                    samples = preprocessor.eoi  # force eoi otken
                    restrict_tokens = 'text'

            if j == model.max_seq_len - 1:
                x[i] = torch.cat((x[i, 1:][None, :], samples), dim=1)
                break

            x[i, j+1] = samples
    return x
