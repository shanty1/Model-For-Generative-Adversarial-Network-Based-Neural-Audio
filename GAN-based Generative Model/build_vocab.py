
import pickle
import argparse
from collections import Counter
import jieba

# nltk.download()

# ============================ 构建词向量 ======================================
class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(path, threshold):
    """Build a simple vocabulary wrapper."""
    # **************************************************************
    counter = Counter()
    text = open(path,'r',encoding='UTF-8').read()
    texts = text.split('\n')
    for i in range(len(texts)):
        # temp = texts[i].replace('.',' ').split(' ')[-1]
        temp = texts[i].replace('，','').replace('.', ' ').split(' ')[-1]
        # print(temp)

        tokens = jieba.lcut(temp)
        # print(tokens)
        counter.update(tokens)

    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    for i, word in enumerate(words):
        vocab.add_word(word)

    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    return vocab


def main(args):
    vocab = build_vocab(path=args.comment_file, threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--comment_file', type=str, 
                        default='./data/comment/comment.txt', 
                        help='path for comment file')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl', 
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=1,
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)