from python_speech_features import mfcc, delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import os
import numpy as np

def audio_mfcc(files,N):
    (rate,sig) = wav.read(files)
    mfcc_feat = mfcc(sig,rate,nfft=1200)
    fbank_feat = logfbank(sig,rate,nfft=1200)
    fbank_feat = delta(fbank_feat, N)

    return fbank_feat

def audio_pad_pack(mfcc_fea,audio_len,mfcc_dim=26):
    if mfcc_fea.shape[0] > audio_len:
        return mfcc_fea[:audio_len]
    else:
        temp = np.zeros([audio_len - mfcc_fea.shape[0],mfcc_dim])
        return np.vstack((mfcc_fea,temp))

def audio_preprocess(path,N,audio_len,mfcc_dim):
    files = os.listdir(path)

    mfcc_features = []
    for audio in files:
        mfcc_features.append(audio_pad_pack(audio_mfcc(path+audio,N),audio_len,mfcc_dim))
    
    return np.array(mfcc_features)

# m = audio_preprocess(PATH,N,AUDIO_LEN,MFCC_DIM)