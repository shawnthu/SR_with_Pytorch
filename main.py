import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import numpy as np
import librosa
from collections import defaultdict


class AudioDst(Dataset):
    def __init__(self, downsample=True):
        data_dir = '/media/shawn/Seagate Expansion Drive/Dataset/Speech/' + 'ST-CMDS-20170001_1-OS/'
        stem_list_path = '/media/shawn/Seagate Expansion Drive/Dataset/Speech/' + 'st_cmds_stem_list.txt'
        text_path = '/media/shawn/Seagate Expansion Drive/Dataset/Speech/st_cmds_text_list.txt'
        stems = open(stem_list_path, 'r').read().split('\n')
        text_list = open(text_path, 'r').read().split('\n')
        text_list = [''.join(ele.strip().split()) for ele in text_list]

        self.data_dir = data_dir
        self.stems = stems
        self.downsample = downsample
        self.word2int, self.int2word = self.make_dict(text_path)
        self.text_list = text_list

    def make_dict(self, text_path, max_word=500):
        dct = defaultdict(int)
        with open(text_path, 'r') as f:
            for line in f:
                text = ''.join(line.strip().split())
                for s in text:
                    dct[s] += 1

        kvs = list(dct.items())
        kvs.sort(key=lambda x: x[1], reverse=True)

        word2int = {'<pad>': 0, '<s>': 1, '</s>': 2, '<unk>': 3}
        word2int.update({ele[0]: i+4  for i, ele in enumerate(kvs[:max_word])})
        int2word = {v: k for k, v in word2int.items()}
        return word2int, int2word

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        # return a tuple: (audio_log_mel, txt)
        full_stem = self.data_dir + self.stems[idx]

        # using librosa
        audio, rate = librosa.load(full_stem + '.wav', sr=None)  # float32, 1D
        feature = librosa.feature.melspectrogram(audio, rate,
                                                 n_fft=int(25*rate/1000),
                                                 hop_length=int(10*rate/1000),
                                                 n_mels=26, fmin=80, fmax=7600)
        feature = np.log(feature)
        if self.downsample:
            feature = feature[:, :int(3 * (feature.shape[1] // 3))]
            feature = feature.T.reshape(feature.shape[1] // 3, -1).astype('float32')
        else:
            feature = feature.T.astype('float32')  # [n_frames, n_mels]

        text = self.text_list[idx]
        text_int = [self.word2int.get(ele, 2) for ele in text]
        return feature, text_int


class BauAttn(nn.Module):
    def __init__(self, enc_size, hidden_size):
        super().__init__()
        attn_size = 64
        self.W_enc = nn.Parameter(data=torch.randn(enc_size, attn_size))
        self.b_attn = nn.Parameter(data=torch.zeros(attn_size))
        self.W_hidden = nn.Parameter(data=torch.randn(hidden_size, attn_size))
        self.v = nn.Parameter(data=torch.randn(attn_size))

        self.reset_params()
        self.attn_size = attn_size

    def reset_params(self):
        nn.init.xavier_normal_(self.W_enc)
        nn.init.xavier_normal_(self.W_hidden)

        nn.init.zeros_(self.b_attn)

    def compute(self, enc_outputs):
        # 在forward之前调用这个
        l, b, d = enc_outputs.shape
        W = self.W_enc.repeat(l, 1, 1)
        parts = torch.bmm(enc_outputs, W) + self.b_attn  # [l, b, d]
        return parts

    def forward(self, enc_outputs, lens, hidden_states, parts=None):
        # enc_outputs: [l, b, d]
        # lens: [b]
        # hidden_state: [b, D]
        # parts: [l, b, d]
        if parts is None:
            parts = self.compute(enc_outputs)
        scores = (torch.tanh(parts + torch.mm(hidden_states, self.W_hidden)) * self.v).sum(dim=2)  # [l, b]
        # print('scores shape:', scores.shape)

        mask = nn.utils.rnn.pad_sequence([torch.ones(l) for l in lens], batch_first=False)
        # mask = (1. - mask) * (-np.inf)  # [l, b] 导致全是nan
        mask.masked_fill_(mask == 0, -np.inf)
        # print('mask:', 'max =', mask.max().item(), 'min =', mask.min().item())

        scores = F.softmax(mask + scores, dim=0)
        # print('scores:', 'max =', scores.max().item(), 'min =', scores.min().item())
        attns = (scores.unsqueeze(dim=2) * enc_outputs).sum(dim=0)  # [b, d]
        return attns


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv1d(input_size, 128, 3)
        # self.relu = nn.ReLU()
        self.hidden_size = 4  # 128
        self.num_layers = 2
        self.lstm = nn.LSTM(input_size=26*3, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, bias=True, batch_first=False,
                            dropout=.0, bidirectional=False)

        self.reset_params()

    def reset_params(self):
        # self.lstm.
        # nn.init.zeros_(self.conv1.bias)
        pass

    def forward(self, x, lens):
        """
        return enc_output
        x: [l, b, d]
        lens: [b]
        """
        # use pack
        bsz = x.size(1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=False)
        h0 = torch.zeros(self.num_layers * 1, bsz, self.hidden_size)
        c0 = torch.zeros(self.num_layers * 1, bsz, self.hidden_size)
        enc_outputs, (h, c) = self.lstm(packed, (h0, c0))
        return enc_outputs

        # # use mask
        # bsz = x.size(1)
        # # h0 = torch.zeros(self.num_layers * 1, bsz, self.hidden_size)
        # # c0 = torch.zeros(self.num_layers * 1, bsz, self.hidden_size)
        # enc_outputs, (h, c) = self.lstm(x, (h, c))
        # return enc_outputs, (h, c)


class Decoder(nn.Module):
    def __init__(self, enc_size):
        super().__init__()

        self.vocab_size = 1024
        self.hidden_size = 128
        self.embed_dim = 64
        self.bsz = 3
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)
        self.cell = nn.LSTMCell(self.embed_dim + enc_size, self.hidden_size)
        self.proj_linear = nn.Linear(self.hidden_size * 2, self.vocab_size)

        self.reset_params()

    def reset_params(self):
        pass

    def forward(self, tokens, attns, h, c):
        # tokes: [b]
        # attns: [b, d]
        x = self.embedding(tokens)  # [b, d]
        input_ = torch.cat((x, attns), dim=1)
        h, c = self.cell(input_, (h, c))
        return h, c


def train():
    # parameters
    pad = 0
    sos = 1
    eos = 2
    unk = 3

    batch_size = 32
    epochs = 1
    num_iterations = 1

    # make data loader
    dst = AudioDst(downsample=True)

    def collate_fn(batch):
        batch.sort(key=lambda x: x[0].shape[0], reverse=True)
        # max_len = batch[0].shape[0]
        t = [torch.from_numpy(ele[0]) for ele in batch]
        lens = torch.IntTensor([ele.size(0) for ele in t])
        padded = nn.utils.rnn.pad_sequence(t, batch_first=False)

        text = [torch.LongTensor(ele[1]) for ele in batch]
        text_lens = torch.IntTensor([ele.size(0) for ele in text])
        text_padded = nn.utils.rnn.pad_sequence(text, batch_first=False)
        return padded, lens, text_padded, text_lens  # [l, b, d], [b], [l, b], [b]

    loader = DataLoader(dst, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, collate_fn=collate_fn)

    encoder = Encoder()
    attn = BauAttn(128, 128)
    decoder = Decoder(encoder.hidden_size)
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(list(encoder.parameters()) +
                           list(attn.parameters()) +
                           list(decoder.parameters()),
                           lr=.0001)

    iteration = 0
    for epoch in range(1, 1+epochs):
        print('EPOCH %d' % epoch)
        for data, lens, text_padded, text_lens in loader:
            enc_outputs = encoder(data, lens)  # packed

            enc_outputs, _ = nn.utils.rnn.pad_packed_sequence(enc_outputs, batch_first=False)  # [l, b, d]
            enc_outputs.sum().backward()
            break
            parts = attn.compute(enc_outputs)
            # print('parts shape:', parts.shape)

            bsz = data.size(1)
            L = text_lens.max().item()
            tokens = torch.full((bsz,), sos, dtype=torch.int64)
            attns = torch.zeros(bsz, encoder.hidden_size)
            h = torch.zeros(bsz, decoder.hidden_size)
            c = torch.zeros(bsz, decoder.hidden_size)
            loss_sum = torch.tensor(0.)
            for l in range(1):
                h, c = decoder(tokens, attns, h, c)  # [b, d]
                d = h
                attns = attn(enc_outputs, lens.tolist(), d, parts)  # [b, d]
                # print('attns: max = %.3f min = %.3f' % (attns.max().item(), attns.min().item()))
                # print('attns shape:', attns.shape)
                logits = decoder.proj_linear(torch.cat((d, attns), dim=1))  # [b, v]
                mask = (text_padded[l] != pad).type(torch.float32)  # [b]
                loss = criterion(logits, text_padded[l])  # [b]
                loss = (loss * mask).sum() / mask.sum()
                loss_sum += loss.mean()
                tokens = text_padded[l]

            loss = loss_sum / L

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iteration += 1
            print('[%3d | epoch %2d] loss=%.3f' % (iteration, epoch, loss.item()))

            if iteration >= num_iterations:
                break


class Search:

    def __init__(self, tgt_dict):
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.scores_buf = None
        self.indices_buf = None
        self.beams_buf = None

    def _init_buffers(self, t):
        if self.scores_buf is None:
            self.scores_buf = t.new()
            self.indices_buf = torch.LongTensor().to(device=t.device)
            self.beams_buf = torch.LongTensor().to(device=t.device)

    def step(self, step, lprobs, scores, beam_size):
        """Take a single search step.

        Args:
            step: the current search step, starting at 0
            lprobs: (bsz x input_beam_size x vocab_size)
                the model's log-probabilities over the vocabulary at the current step
            scores: (bsz x input_beam_size x step)
                the historical model scores of each hypothesis up to this point

        Return: A tuple of (scores, indices, beams) where:
            scores: (bsz x output_beam_size)
                the scores of the chosen elements; output_beam_size can be
                larger than input_beam_size, e.g., we may return
                2*input_beam_size to account for EOS
            indices: (bsz x output_beam_size)
                the indices of the chosen elements
            beams: (bsz x output_beam_size)
                the hypothesis ids of the chosen elements, in the range [0, input_beam_size)
        """
        raise NotImplementedError

    def set_src_lengths(self, src_lengths):
        self.src_lengths = src_lengths


class BeamSearch(Search):

    def __init__(self, tgt_dict):
        super().__init__(tgt_dict)

    def step(self, step, lprobs, scores):
        super()._init_buffers(lprobs)
        bsz, beam_size, vocab_size = lprobs.size()

        if step == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            lprobs = lprobs[:, ::beam_size, :].contiguous()
        else:
            # make probs contain cumulative scores for each hypothesis
            lprobs.add_(scores[:, :, step - 1].unsqueeze(-1))

        torch.topk(
            lprobs.view(bsz, -1),
            k=min(
                # Take the best 2 x beam_size predictions. We'll choose the first
                # beam_size of these which don't predict eos to continue with.
                beam_size * 2,
                lprobs.view(bsz, -1).size(1) - 1,  # -1 so we never select pad
            ),
            out=(self.scores_buf, self.indices_buf),
        )
        torch.div(self.indices_buf, vocab_size, out=self.beams_buf)
        self.indices_buf.fmod_(vocab_size)
        return self.scores_buf, self.indices_buf, self.beams_buf


def test():
    dst = AudioDst(downsample=True)
    print('length:', len(dst))
    # y = dst[1234]
    # print(y.shape, y.max(), y.min(), y.dtype)

    # loader = DataLoader(dst, batch_size=4, shuffle=True, num_workers=0, pin_memory=False)

    def collate_fn(batch):
        batch.sort(key=lambda x: x.shape[0], reverse=True)
        # max_len = batch[0].shape[0]
        t = [torch.from_numpy(ele) for ele in batch]
        lens = torch.IntTensor([ele.size(0) for ele in t])
        padded = nn.utils.rnn.pad_sequence(t, batch_first=False)
        return padded, lens  # [l, b, d]

    encoder = Encoder()
    loader = DataLoader(dst, batch_size=4, shuffle=False, num_workers=0, pin_memory=False, collate_fn=collate_fn)

    for ele, lens in loader:
        print(ele.shape, ele.max(), ele.min(), ele.dtype)
        print('lens:', lens)

        y = encoder(ele, lens)
        print(y)  # packedsequence就是以时间轴，每次取一个'batch'，每个step的batch size一般是递减的
        # print(y.shape, y.dtype)
        break


# test()
train()