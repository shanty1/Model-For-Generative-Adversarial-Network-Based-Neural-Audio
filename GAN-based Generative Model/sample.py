import torch
import numpy as np
import argparse
import pickle
import os
from torchvision import transforms
from build_vocab import Vocabulary
import matplotlib.pyplot as plt
from model import EncoderRNN, DecoderRNN
from audio_pre import audio_pad_pack, audio_mfcc
import numpy as np
from torch.autograd import Variable
import random
from data_loader import caculate_max_len
from data_loader import data_get
from data_loader import get_audio_fea
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import  time

AUDIO_LEN = 1000
N = 9
MFCC_DIM = 39
Z_DIM = 20#16

x=[]
y=[]

def load_audio(audio):
    mfcc_fea = audio_mfcc(audio,N)
    mfcc_fea = audio_pad_pack(mfcc_fea,AUDIO_LEN,MFCC_DIM)
    return mfcc_fea

def main(args):
    # random set
    manualSeed = random.randint(1, 100)
    # print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # mfcc_features = audio_preprocess(args.audio_dir, N, AUDIO_LEN, MFCC_DIM).astype(np.float32)



    # Build models
    encoder = EncoderRNN(MFCC_DIM, args.embed_size, args.hidden_size).to(device)
    decoder = DecoderRNN(args.embed_size+Z_DIM, args.hidden_size, len(vocab), args.num_layers)
    # decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    
    for i in [1,2,13,14,28,38,20,87,77,28,10,36,100,63,45]:
        x.append(i)
        args.audio='./data/audio/swz{}.wav'.format(i)
        # Prepare an image

        start = time.perf_counter()
        audio = get_audio_fea(args.audio)
        audio = torch.tensor(audio).unsqueeze(0).to(device)

        # Generate an caption from the image
        feature = encoder(audio,[audio.shape[1]])
        if (Z_DIM > 0):
            z = Variable(torch.randn(feature.shape[0], Z_DIM)).cuda()
            feature = torch.cat([z,feature],1)
        sampled_ids = decoder.sample(feature)
        sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)

        # Convert word_ids to words
        sampled_caption = []
        for word_id in sampled_ids:
            word = vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sampled_caption)

        # Print out the image and the generated caption
        sentence = sentence.replace('<start>','').replace('<unk>','，').replace('<end>','').replace(' ','')
        # print(sentence)

        end = time.perf_counter()
        y.append(end - start)

if __name__ == '__main__':
    # for i in range(1,101):

    i = 1

    parser = argparse.ArgumentParser()
    # print(i,end='. ')
    parser.add_argument('--audio', type=str, default='./data/audio/swz{}.wav'.format(i), help='test audio path') #110-30  300-30-15   300-90(90-1全拟合)
    parser.add_argument('--encoder_path', type=str, default='./models/encoder-328-180.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='./models/decoder-328-180.ckpt', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')

    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    args = parser.parse_args()
    main(args)

    plt.plot(sorted(x), y,'r')
    plt.show()

    i = 0
    while i < len(x):
        print(x[i],'\t',y[i])
        i+=1