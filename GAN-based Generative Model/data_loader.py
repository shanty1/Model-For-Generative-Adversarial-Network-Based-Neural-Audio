import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import jieba
from build_vocab import Vocabulary
from python_speech_features import mfcc, delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav


def caculate_max_len(audio_dir,text_file,vocab):
    mfcc_features = []
    comment_features = []
    audio_files = os.listdir(audio_dir)
    for audio in audio_files:
        mfcc_features.append(get_audio_fea(audio_dir + audio))
    document_text = open(text_file, 'r',encoding='UTF-8').read()
    comment_list = document_text.split('\n')
    for comment in comment_list:
        comment_features.append(get_comment_fea(comment,vocab))
    mfcc_lengths = [len(item) for item in mfcc_features]
    comment_lengths = [len(item) for item in comment_features]
    return max(mfcc_lengths),max(comment_lengths),mfcc_features[0].shape[1]
 

def get_audio_fea(audio_file, audio_max_len):
    audio_mfcc = get_audio_fea(audio_file)
    # 1. 语音填充0
    audio_len = audio_mfcc.shape[0]
    mfcc_zero = np.zeros([audio_max_len - audio_len, audio_mfcc.shape[1]])
    audio_mfcc = torch.from_numpy(np.vstack((audio_mfcc, mfcc_zero)).astype(np.float32))
    return mfcc,audio_len
    
def get_audio_fea(audio_file):
    (rate,sig) = wav.read(audio_file)
    wav_feature = mfcc(sig, rate,nfft=1200)
    d_mfcc_feat1 = delta(wav_feature, 1)
    d_mfcc_feat2 = delta(wav_feature, 2)
    wav_feature = np.hstack((wav_feature, d_mfcc_feat1, d_mfcc_feat2))
    return wav_feature.astype(np.float32)

def get_comment_fea(comment,vocab):
    content = comment[comment.index(' ')+1:]
    tokens = jieba.lcut(content)
    comment_codes = []
    comment_codes.append(vocab('<start>'))
    comment_codes.extend([vocab(token) for token in tokens])
    comment_codes.append(vocab('<end>'))
    return comment_codes

def data_get(audio_dir,audio_max_len, text_path, comment_max_len, vocab):
    text = open(text_path, 'r',encoding='UTF-8').read()
    texts = text.split('\n')
    data_list = []
    for comment in texts:
        # 编号，如 1.1
        i = comment[:comment.index(' ')]
        if( int(i[:i.index('.')])>90):
            continue
        # 评论文本内容
        content = comment[comment.index(' ')+1:]
        print(i,'：',   content)

        # 1.获取评论编码
        comment_codes = get_comment_fea(comment,vocab)
        comment_codes = np.array(comment_codes)
        # 2.获取评论对应的语音编码
        audio_file_path = audio_dir + os.path.sep + 'swz{}.wav'.format(i[:i.index('.')])
        audio_mfcc = get_audio_fea(audio_file_path)
        # audio_mfcc = np.array(audio_mfcc)

        # 1. 评论填充0
        comment_len = comment_codes.shape[0]
        comment_zero = np.zeros(comment_max_len - comment_len)
        comment_codes = torch.from_numpy(np.concatenate([comment_codes, comment_zero])).long()
        
        # 1. 语音填充0
        audio_len = audio_mfcc.shape[0]
        mfcc_zero = np.zeros([audio_max_len - audio_len, audio_mfcc.shape[1]])
        audio_mfcc = torch.from_numpy(np.vstack((audio_mfcc, mfcc_zero)).astype(np.float32))

        comment = (comment_codes, comment_len)
        audio = (audio_mfcc, audio_len)
        data = (audio,comment)
        data_list.append(data)
    return data_list