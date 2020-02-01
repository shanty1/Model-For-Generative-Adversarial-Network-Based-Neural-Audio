import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os
import pickle
 
from data_loader import caculate_max_len 
from data_loader import data_get 
from build_vocab import Vocabulary
from model import EncoderRNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from audio_pre import audio_preprocess
import random

 
Z_DIM = 20 #20

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
def main(args):

    # random set
    manualSeed = random.randint(1, 100)
    # print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    audio_len, comment_len, mfcc_dim = caculate_max_len(args.audio_dir,args.text_path, vocab)
    # mfcc_features = audio_preprocess(args.audio_dir, N, AUDIO_LEN, MFCC_DIM).astype(np.float32)
    
    # Build data loader
    data_loader = data_get(args.audio_dir,audio_len, args.text_path, comment_len, vocab )

    # Build the models
    encoder = EncoderRNN(mfcc_dim, args.embed_size, args.hidden_size).to(device)
    decoder = DecoderRNN(args.embed_size+Z_DIM, args.hidden_size, len(vocab), args.num_layers).to(device)
    # decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
 
    
    # Train the models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        for i, ((audio,audio_len), (comment,comment_len)) in enumerate(data_loader):
            audio = audio.to(device)
            audio = audio.unsqueeze(0)
            comment = comment.to(device)
            comment = comment.unsqueeze(0)

            targets = pack_padded_sequence(comment, [comment_len], batch_first=True)[0]
            
            # Forward, backward and optimize
            audio_features = encoder(audio, [audio_len])
            if(Z_DIM>0):
                z = Variable(torch.randn(audio_features.shape[0], Z_DIM)).cuda()
                audio_features = torch.cat([z,audio_features],1)
            outputs = decoder(audio_features, comment, [comment_len])
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'.format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item())))
                
            # Save the model checkpoints
        if (epoch+1) % args.save_step == 0:
            torch.save(decoder.state_dict(), os.path.join(
                args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
            torch.save(encoder.state_dict(), os.path.join(
                args.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--audio_dir', type=str, default='./data/audio/', help='directory for audioes')
    parser.add_argument('--text_path', type=str, default='./data/comment/comment.txt', help='path for comment text')
    parser.add_argument('--log_step', type=int , default=1, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1, help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)