import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, dataset, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()
        self.seq_len = seq_len

        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # Special tokens for SOURCE tokenizer
        self.src_sos_token_id = tokenizer_src.token_to_id("[SOS]")
        self.src_eos_token_id = tokenizer_src.token_to_id("[EOS]")
        self.src_pad_token_id = tokenizer_src.token_to_id("[PAD]")
        
        # Special tokens for TARGET tokenizer
        self.tgt_sos_token_id = tokenizer_tgt.token_to_id("[SOS]")
        self.tgt_eos_token_id = tokenizer_tgt.token_to_id("[EOS]")
        self.tgt_pad_token_id = tokenizer_tgt.token_to_id("[PAD]")

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        """
        Fetches corresponding sentence pair --> Tokenizer --> Add Special Tokens --> Padding --> Prepare Mask

        Construct Tensors:
            Encoder Input Format:
            - [SOS] + Source_Token_ids + [EOS] + [PAD]s
            - Length = seq_len

            Decoder Input Format:
            - [SOS] + Target_Token_ids + [PAD]s (The model learns to predict [EOS])
            - Length = seq_len

            Label:
            - Target_Token_ids + [EOS] + [PAD]s
            - Length = seq_len

        Output:
            - encoder_input
            - decoder_input
            - encoder_mask
            - decoder_mask
            - label
            - src_text
            - tgt_text
        """
        # Fetch
        src_target_pair = self.dataset[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Tokenize
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Padding
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # SOS, EOS
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # SOS

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')
        
        # Add SOS and EOS to the source text
        encoder_input = torch.cat(
            [
                torch.tensor([self.src_sos_token_id], dtype=torch.int64),
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                torch.tensor([self.src_eos_token_id], dtype=torch.int64),
                torch.tensor([self.src_pad_token_id] * enc_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # Add SOS to the decoder input
        decoder_input = torch.cat(
            [
                torch.tensor([self.tgt_sos_token_id], dtype=torch.int64),
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.tgt_pad_token_id] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # Add EOS to the label (Decoder output)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.tgt_eos_token_id], dtype=torch.int64),
                torch.tensor([self.tgt_pad_token_id] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            'encoder_input': encoder_input, # (seq_len)
            'decoder_input': decoder_input, # (seq_len)
            'encoder_mask': (encoder_input != self.src_pad_token_id).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            'decoder_mask': (decoder_input != self.tgt_pad_token_id).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, 1, seq_len) & (1, seq_len, seq_len)
            'label': label, # (seq_len)
            'src_text': src_text,
            'tgt_text': tgt_text
        }
    
def causal_mask(size):
    '''
    In transformer decoder, when processing a target token, only restrict it to look at previous tokens in target sequence, not future ones
    '''
    # Upper triangular matrix, result in a matrix where positions "future" tokens are 1, other are zero
    mask = torch.triu(torch.ones(1, size, size), diagonal = 1).type(torch.int)
    return mask == 0 # Reverse