from Config import *
from Dataset import *
from tqdm import tqdm

# Training
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

def get_all_sentences(dataset, lang):
    for item in dataset:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, dataset, lang):
    # Path where tokenizer for given language is saved
    tokenizer_path = Path(config['tokenizer_file'].format(lang))

    if not Path.exists(tokenizer_path):
        # Build 
        print("=== Tokenizer: Building ===")
        tokenizer = Tokenizer(WordLevel(unk_token = "[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()

        # Train
        trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency = 2) # Words appearing less than twice will become "[UNK]"
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer = trainer)

        # Save
        tokenizer.save(str(tokenizer_path))
    else:
        # Load
        print("=== Tokenizer: Loading ===")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer

def split_tokens_into_chunks(token_ids: list[int], max_len: int) -> list[list[int]]:
    """Splits a list of token IDs into chunks of a maximum length."""
    chunks = []
    if not token_ids: # Handle empty list of tokens
        return []
    if max_len <= 0: # Ensure max_len is positive
        # This case should ideally be prevented by checks before calling
        return [token_ids] if token_ids else [] 
    for i in range(0, len(token_ids), max_len):
        chunks.append(token_ids[i:i + max_len])
    return chunks

def chunk(original_dataset, tokenizer_src, tokenizer_tgt, config):
    """
    Processes the original dataset by chunking long sentences.
    Returns a new list of items, where each item is a dictionary 
    {'translation': {src_lang: text, tgt_lang: text}},
    suitable for BilingualDataset.
    """
    processed_items = []
    src_lang = config['lang_src']
    tgt_lang = config['lang_tgt']
    seq_len = config['seq_len']

    # Maximum number of actual tokens allowed in a chunk for source and target
    # Source: seq_len - 2 (to accommodate SOS and EOS tokens added by BilingualDataset)
    # Target: seq_len - 1 (to accommodate SOS for decoder_input, or EOS for label, added by BilingualDataset)
    max_src_tokens_per_chunk = seq_len - 2
    max_tgt_tokens_per_chunk = seq_len - 1

    num_original_pairs = 0
    num_chunk_pairs_created = 0
    num_original_pairs_fully_discarded = 0
    num_potential_chunk_pairs_discarded_empty_text = 0

    for item in tqdm(original_dataset, desc="Chunking dataset"):
        num_original_pairs += 1
        src_text_original = item['translation'][src_lang]
        tgt_text_original = item['translation'][tgt_lang]

        src_token_ids = tokenizer_src.encode(src_text_original).ids
        tgt_token_ids = tokenizer_tgt.encode(tgt_text_original).ids
        
        item_produced_valid_chunk = False

        if len(src_token_ids) <= max_src_tokens_per_chunk and len(tgt_token_ids) <= max_tgt_tokens_per_chunk:
            # Sentence pair is short enough, add as is if text is not empty/whitespace
            if src_text_original.strip() and tgt_text_original.strip():
                processed_items.append({'translation': {src_lang: src_text_original, tgt_lang: tgt_text_original}})
                num_chunk_pairs_created += 1
                item_produced_valid_chunk = True
            else:
                num_potential_chunk_pairs_discarded_empty_text += 1 # technically not a chunk pair yet
        else:
            # One or both sentences are too long, chunk them
            src_token_chunks = split_tokens_into_chunks(src_token_ids, max_src_tokens_per_chunk)
            tgt_token_chunks = split_tokens_into_chunks(tgt_token_ids, max_tgt_tokens_per_chunk)

            # Pair up the chunks: take the minimum number of chunks from source and target
            num_chunks_to_pair = min(len(src_token_chunks), len(tgt_token_chunks))

            for i in range(num_chunks_to_pair):
                current_src_chunk_ids = src_token_chunks[i]
                current_tgt_chunk_ids = tgt_token_chunks[i]

                if not current_src_chunk_ids and not current_tgt_chunk_ids: # both empty, skip
                    num_potential_chunk_pairs_discarded_empty_text +=1
                    continue

                src_chunk_text = tokenizer_src.decode(current_src_chunk_ids)
                tgt_chunk_text = tokenizer_tgt.decode(current_tgt_chunk_ids)
                
                # Add chunk pair if both decoded texts are non-empty/whitespace
                if src_chunk_text.strip() and tgt_chunk_text.strip():
                    processed_items.append({'translation': {src_lang: src_chunk_text, tgt_lang: tgt_chunk_text}})
                    num_chunk_pairs_created += 1
                    item_produced_valid_chunk = True
                else:
                    num_potential_chunk_pairs_discarded_empty_text += 1
        
        if not item_produced_valid_chunk:
            num_original_pairs_fully_discarded +=1

    print(f"Finished processing and chunking.")
    print(f"Number of original sentence pairs: {num_original_pairs}.")
    print(f"Total usable sentence/chunk pairs created: {len(processed_items)}.")

    if not processed_items:
        raise ValueError("Processing and chunking resulted in an empty dataset. Check seq_len, tokenizer, and data quality. Ensure original data is not all empty strings.")

    return processed_items

def get_dataset(config, split = "train"):
    dataset_raw = load_dataset(
        f"{config['datasource']}",
        f"{config['lang_src']}-{config['lang_tgt']}",
        split = split
    ).shuffle(seed = 42)
    # dataset_raw = load_dataset(
    #     f"{config['datasource']}",
    #     f"iwslt2015-{config['lang_src']}-{config['lang_tgt']}", 
    #     revision = "refs/convert/parquet",
    #     split = split
    # ).shuffle(seed = 42)

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, dataset_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, dataset_raw, config['lang_tgt'])

    # Process and chunk dataset
    dataset_raw = chunk(dataset_raw, tokenizer_src, tokenizer_tgt, config)

    # Check max length
    max_len_src = 0
    max_len_tgt = 0

    for item in dataset_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    print(f'Length of source tokenizer: {tokenizer_src.get_vocab_size()}')
    print(f'Length of target tokenizer: {tokenizer_tgt.get_vocab_size()}')

    # Size ratio (Training, Validation) = (0.9, 0.1)
    train_ds_size = int(0.9 * len(dataset_raw))
    val_ds_size = len(dataset_raw) - train_ds_size
    train_dataset_raw, val_dataset_raw = random_split(dataset_raw, [train_ds_size, val_ds_size])

    train_dataset = BilingualDataset(train_dataset_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_dataset = BilingualDataset(val_dataset_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size = config['batch_size'], shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = 1, shuffle = True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt