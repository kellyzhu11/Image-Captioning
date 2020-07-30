from pathlib import Path
import json
import argparse
import numpy as np 
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from tqdm import tqdm


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    ## Create model directory
    #if not os.path.exists(args.model_path):
    #    os.makedirs(args.model_path)
    
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.Resize([256, 256]), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers, dropout = args.dropout)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    # Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab, 
                             transform, args.batch_size,
                             shuffle=False, num_workers=args.num_workers) 

    # Test the models
    total_step = len(data_loader)
    start_token = vocab('<start>')
    syn_captions = []
    keys = {}
    
    for i, (images, captions, img_ids, lengths) in tqdm(enumerate(data_loader), total=total_step, leave=False, ncols=80, unit='b'):
        
        # Set mini-batch dataset
        images = images.to(device)

        # Generate an caption from the image
        feature = encoder(images)
        if args.seach_algorithm == 'beam_search':
            sampled_ids = decoder.beam_search(feature, start_token, beam_width = args.beam_width)
        elif args.seach_algorithm == 'greedy_search':
            sampled_ids = decoder.greedy_search(feature)
    
        sampled_ids = sampled_ids.cpu().numpy()
        
        for j in range(sampled_ids.shape[0]):
            sampled_caption = []
            sampled_id = sampled_ids[j]
            img_id = img_ids[j]

            for word_id in sampled_id:
                word = vocab.idx2word[word_id]
                sampled_caption.append(word)
                if word == '<end>':
                    break
            
            sentence = ' '.join(sampled_caption)
            # only keep one caption for each image
            try:
                _ = keys[img_id]
            except KeyError:
                keys[img_id] = 1
                item = {'image_id': img_id,
                        'caption': sentence}
                syn_captions.append(item)

    output_dir = Path(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = output_dir / Path('predictions.json')
    with open(str(output_path), 'w') as fout:
        json.dump(syn_captions, fout)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='data/val2014', help='directory for validation images')
    parser.add_argument('--caption_path', type=str, default='data/annotations/captions_val2014.json', help='path for validation annotation json file')
    parser.add_argument('--encoder_path', type=str, default='models/encoder-20-1000.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/decoder-20-1000.ckpt', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--output_dir', type=str, default='./results', help='path for test output json file')
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')

    parser.add_argument('--batch_size', type=int, default=20, help='number of batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers for data loader')
    parser.add_argument('--dropout', type=int, default=0)
    parser.add_argument('--seach_algorithm', type=str, default='beam_search')
    parser.add_argument('--beam_width', type=int, default=3)

    args = parser.parse_args()
    print(args)
    main(args)