"""Byte-level BPE fitted only to supplied training question/code pairs."""
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

PAD, BOS, EOS, UNK = range(4)


def train_tokenizer(texts, path: Path, vocab_size=2048):
    tok = Tokenizer(models.BPE(unk_token='<unk>'))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=vocab_size,
        special_tokens=['<pad>', '<bos>', '<eos>', '<unk>'],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    tok.save(str(path))
    return tok


def encode(tok, text, max_length):
    return tok.encode(text).ids[:max_length-1] + [EOS]


def decode(tok, ids):
    ids = list(ids)
    if EOS in ids: ids = ids[:ids.index(EOS)]
    return tok.decode(ids, skip_special_tokens=True)
