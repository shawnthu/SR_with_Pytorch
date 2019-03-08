import os
from time import time
from pathlib import Path
from glob import glob

from torch.utils.data import Dataset, DataLoader

import numpy as np


def st_cmds():
    data_dir = '/media/shawn/Seagate Expansion Drive/Dataset/Speech/ST-CMDS-20170001_1-OS/'
    # x = os.listdir(data_dir)
    # x = glob(data_dir + '*.wav')
    # print(len(x))
    # x = [Path(ele).stem for ele in x]
    # print(x[:10])
    # print(x[-10:])

    # write_path = '/media/shawn/Seagate Expansion Drive/Dataset/Speech/' + 'st_cmds_stem_list.txt'
    # open(write_path, 'w').write('\n'.join(x))

    stem_list_path = '/media/shawn/Seagate Expansion Drive/Dataset/Speech/st_cmds_stem_list.txt'
    stems = open(stem_list_path, 'r').read().split('\n')
    print(stems[:5])
    ts = time()
    texts = [open(data_dir + stem + '.txt').read() for stem in stems]
    print('time cost %.3fs' % (time()-ts))

    text_list_write_path = '/media/shawn/Seagate Expansion Drive/Dataset/Speech/' + 'st_cmds_text_list.txt'
    open(text_list_write_path, 'w').write('\n'.join(texts))


def feature():
    from python_speech_features import logfbank, mfcc
    from scipy.io.wavfile import read as wav_read
    import librosa

    # path = '/media/shawn/Seagate Expansion Drive/Dataset/Speech/ST-CMDS-20170001_1-OS/20170001P00241I0075.wav'
    # path = '/media/shawn/Seagate Expansion Drive/Dataset/Speech/ST-CMDS-20170001_1-OS/20170001P00001A0001.wav'
    path = '/media/shawn/Seagate Expansion Drive/Dataset/Speech/ST-CMDS-20170001_1-OS/20170001P00212I0045.wav'

    # rate, audio = wav_read(path)

    # librosa official example
    ts = time()
    audio, rate = librosa.load(path, None)
    print('rate:', rate,
          '\naudio:', audio.shape, audio.dtype, audio.max(), audio.min())

    # feat = librosa.feature.mfcc(audio, rate)
    # in librosa: window size <=> n_fft; window stride <=> hop_length
    feat = librosa.feature.melspectrogram(audio, rate, n_fft=int(25*rate/1000),
                                          hop_length=int(10*rate/1000),
                                          fmin=80, fmax=7600)

    feat = np.log(feat).astype('float32')
    print(feat.shape, feat.dtype, feat.max(), feat.min())
    print('time cost %.3fs' % (time()-ts))

    # official example
    # mfcc_feat = mfcc(audio, rate)
    # fbank_feat = logfbank(audio, rate)

    # normalize the raw audio data int16
    # audio = audio.astype('float32')
    # audio = audio - audio.mean()
    # audio = audio / np.abs(audio).max()
    # # print(audio.shape, audio.max(), audio.min(), audio.dtype)
    #
    # audio = audio / (2 ** 15 - 1)
    # # print('audio dype:', audio.dtype, audio.max(), audio.min())
    #
    # # frame_length = 25 * rate / 1000
    # # nfft = int(2 ** np.ceil(np.log2(frame_length)))
    # # print('nfft:', nfft)  # when rate = 16000, then nfft = 512
    #
    # # print('rate:', rate)
    #
    # log_mel = logfbank(audio.reshape(-1, 1), rate, .025, .01,
    #                    nfilt=26, nfft=512, lowfreq=80, highfreq=7600, preemph=.97)
    # print('log mel shape:', log_mel.shape)
    # return log_mel


def dataset():
    class MyDst(Dataset):
        def __init__(self):
            self.value = [4, 20, 12]

        def __len__(self):
            return len(self.value)

        def __getitem__(self, idx):
            return self.value[idx]

    dst = MyDst()
    print('length:', len(dst))
    for ele in dst[:2]:
        print(ele)


# st_cmds()
feature()
# dataset()