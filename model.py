import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, dropout, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout = dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim = 2)

        self.hidden_size = hidden_size
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)

        outputs = self.linear(hiddens[0])
        return outputs
    
    def greedy_search(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)

        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size) #states:((1, batch, hidden), (1, batch, hidden))
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids
    
    def beam_search(self, features, start_token: int, beam_width, states=None):
        inputs = features.unsqueeze(1)
        batch_size = inputs.size()[0]

        hidden, states = self.lstm(inputs, states)
        words = Variable(torch.Tensor([start_token]).long(), requires_grad=False).repeat(batch_size).view(batch_size, 1, 1) #[batch_size, length:1, beam:1]
        probs = Variable(torch.zeros(batch_size, 1))  # [batch_size, beam:1]
        if torch.cuda.is_available():
            words = words.cuda()
            probs = probs.cuda()
        h, c = states #[1, batch_size, hidden_size]

        all_hidden = h.unsqueeze(3)  # [1, batch_size, hidden_size, beam:1]
        all_cell = c.unsqueeze(3)  # [1, batch_size, hidden_size, beam:1]
        all_words = words  # [batch_size, length:1, beam:1]
        all_probs = probs  # [batch_size, beam:1]
        for t in range(self.max_seg_length):
            new_words = []
            new_cell = []
            new_hidden = []
            new_probs = []
            tmp_words = all_words.split(1, 2)
            tmp_probs = all_probs.split(1, 1)
            tmp_hidden = all_hidden.split(1, 3)
            tmp_cell = all_cell.split(1, 3)
            for i in range(len(tmp_words)):
                last_word = tmp_words[i].split(1, 1)[-1].view(batch_size) #[batch_size,]
                inputs = self.embed(last_word).view(batch_size,1, -1).contiguous() #[batch_size, 1,embed_size]
                last_state = (tmp_hidden[i].squeeze(3).contiguous(), tmp_cell[i].squeeze(3).contiguous()) #[1, batch_size, hidden_size]
                hidden, states = self.lstm(inputs, last_state)

                probs = self.log_softmax(self.linear(hidden))  # [batch, 1, vocab_size]

                probs, indices = probs.topk(beam_width, 2) #indices [batch, 1, beam_width]
                probs = probs.view(batch_size, beam_width)  # [batch, beam_width]

                tmp_words_rep = tmp_words[i].repeat(1, 1, beam_width)

                probs_cand = tmp_probs[i] + probs  # [batch, beam_width]
                words_cand = torch.cat([tmp_words_rep, indices], 1)  # [batch, length+1, beam]
                hidden_cand = states[0].unsqueeze(3).repeat(1, 1, 1, beam_width)  # [1, batch, lstm_dim, beam]
                cell_cand = states[1].unsqueeze(3).repeat(1, 1, 1, beam_width)

                new_words.append(words_cand)
                new_probs.append(probs_cand)
                new_hidden.append(hidden_cand)
                new_cell.append(cell_cand)

            new_words = torch.cat(new_words, 2)  # [batch, length+1, beam*beam]
            new_probs = torch.cat(new_probs, 1)  # [batch, beam*beam]
            new_cell = torch.cat(new_cell, 3)  # [1, batch, lstm_dim, beam*beam]
            new_hidden = torch.cat(new_hidden, 3)  # [1, batch, lstm_dim, beam*beam]

            probs, idx = new_probs.topk(beam_width, 1)  # [batch, beam]
            idx_words = idx.view(batch_size, 1, beam_width)
            idx_words = idx_words.repeat(1, t+2, 1)
            idx_states = idx.view(1, batch_size, 1, beam_width).repeat(1, 1, self.hidden_size, 1)

            # reduce the beam*beam candidates to top@beam candidates
            all_probs = probs
            all_words = new_words.gather(2, idx_words)
            all_hidden = new_hidden.gather(3, idx_states)
            all_cell = new_cell.gather(3, idx_states)

        idx = all_probs.argmax(1)  # [batch]
        idx = idx.view(batch_size, 1, 1).repeat(1, self.max_seg_length+1, 1)
        sampled_ids = all_words.gather(2, idx).squeeze(2)  # [batch, length]

        return sampled_ids