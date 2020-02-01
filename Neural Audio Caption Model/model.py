import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_LAYER = 1

class EncoderRNN(nn.Module):
    def __init__(self, input_SIZE, embed_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.rnn = nn.GRU(input_SIZE, hidden_size, NUM_LAYER, batch_first=True, bidirectional=True)
        # self.linear = nn.Linear(hidden_size*seq_num, embed_size)
        self.linear = nn.Linear(hidden_size*2, embed_size)
        # self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, audioes, len):
        audioes = pack_padded_sequence(audioes, len, batch_first=True)
        out, h = self.rnn(audioes)
        out = pad_packed_sequence(out, batch_first=False)
        len = out[1][0]
        out = out[0][len-1]

        # out = self.linear(out)
        out = self.linear(torch.cat((h[0],h[1]),dim=1))
        # embedding = self.bn(embedding)
        return out

# class EncoderCNN(nn.Module):
#     def __init__(self, embed_size):
#         """Load the pretrained ResNet-50 and replace top fc layer."""w
#         super(EncoderCNN, self).__init__()
#         resnet = models.resnet50(pretrained=False)
#         modules = list(resnet.children())[:-1]      # delete the last fc layer.
#         self.resnet = nn.Sequential(*modules)
#         self.linear = nn.Linear(resnet.fc.in_features, embed_size)
#         self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
#     def forward(self, images):
#         """Extract feature vectors from input images."""
#         with torch.no_grad():
#             features = self.resnet(images)
#         features = features.reshape(features.size(0), -1)
#         features = self.bn(self.linear(features))
#         return features

# class EncoderCNN(nn.Module):
#     def __init__(self, input_SIZE, embed_size, hidden_size):
#         super(EncoderCNN, self).__init__()
#         self.input = input_SIZE
#         self.lstm = nn.LSTM(self.input,hidden_size,NUM_LAYER,batch_first=True)
#         # self.linear = nn.Linear(hidden_size*seq_num, embed_size)
#         self.linear = nn.Linear(hidden_size, embed_size)
#         # self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
#     def forward(self, audioes):
#         embedding,_ = self.lstm(audioes)
#         # embedding = embedding[-1].contiguous().view(embedding[-1].shape[0],-1)
#         embedding = embedding.permute(1,0,2)
#         embedding = self.linear(embedding[-1])
#         # embedding = self.bn(embedding)
#         return embedding


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=50):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True,dropout=0.2)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        # packed = pack_padded_sequence(features.expand(lengths[0],features.shape[1]).unsqueeze(0), lengths, batch_first=True)
        out, _ = self.lstm(packed)
        out = pad_packed_sequence(out, batch_first=True)[0][0]
        outputs = self.linear(out)
        return outputs 
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1).to(device)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids