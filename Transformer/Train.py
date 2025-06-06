from Model import build_transformer
from Dataset import BilingualDataset, causal_mask
from Config import get_config, get_weights_file_path, latest_weights_file_path
from Tokenizer import get_all_sentences, get_or_build_tokenizer, get_dataset

# Dataset Helpers
from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

# Utils
import warnings
from tqdm import tqdm
import os
from pathlib import Path

# Huggingface's Datasets and Tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

# Evaluation
import torchmetrics
from torch.utils.tensorboard import SummaryWriter

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    """
    Performs translation for a single source sentence using greedy decoding strategy

    During validation, use it to check performance
    """
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Source sentence is passed through encoder, then it will be reused for each step of decoding target sentence
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with SOS token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        # Build mask for current target (decoder_input)
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Get decoder_output based on encoder_output and decoder_input
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask) # (batch_size, current_target_seq_len, d_model)

        # Get next token

        # Get probabilities for the final word (Word being generated)
        prob = model.project(out[:, -1])  # out[:, -1]: (batch_size, d_model) --> prob: (batch_size, target_vocab_size)
        _, next_word = torch.max(prob, dim = 1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)],
            dim = 1
        ) # Append the predicted next_word into decoder_input for further usages
        
        if next_word == eos_idx: # Stop if [EOS] predicted
            break
    
    return decoder_input.squeeze(0) # Remove batch dimension

def run_validation(model, val_dataset, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples = 2):
    """
    Evaluate model on validation set
    """
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    # Get console window width
    try:
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        console_width = 80

    with torch.no_grad():
        for batch in val_dataset:
            count += 1
            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (b, 1, 1, seq_len)

            # Check if batch size == 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # Print source, target, model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break

    # Evaluate the character error rate
    if writer:
        # Compute the char error rate
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)   
        writer.add_scalar('Validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('Validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('Validation BLEU', bleu, global_step)
        writer.flush()
    
def get_model(config, vocab_src_len, vocab_tgt_len):
    print("Current model source tokenizer:", vocab_src_len)
    print("Current model target tokenizer:", vocab_tgt_len)
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], d_model = config['d_model'], N = 2)
    return model

def train_model(config):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.has_mps or torch.backends.mps.is_available() else 'cpu'
    print('Using device:', device)

    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")

    device = torch.device(device)

    # Make sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents = True, exist_ok = True)

    # Load Dataset
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    
    # Build model
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps = 1e-9)

    # If the user specified a model to preload before training, load
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print("No model to preload, starting from scratch")

    loss_fn = nn.CrossEntropyLoss(ignore_index = tokenizer_tgt.token_to_id('[PAD]'), label_smoothing = 0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc = f"Processing Epoch {epoch:02d}")

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('Train loss', loss.item(), global_step)
            writer.flush()

            # Back prop
            loss.backward()

            # Update weights
            optimizer.step()
            optimizer.zero_grad(set_to_none = True)

            global_step += 1
        
        # Run validation at the end of every epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)