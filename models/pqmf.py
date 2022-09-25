'''
@File    :   subband_util.py
@Contact :   liu.8948@buckeyemail.osu.edu
@License :   (C)Copyright 2020-2021
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/4/3 4:54 PM   Haohe Liu      1.0         None
'''
import sys
sys.path.append("/")
import torch
import torch.nn as nn
import numpy as np
import os.path as op
from scipy.io import loadmat


def load_mat2numpy(fname=""):
    '''
    Args:
        fname: pth to mat
        type:
    Returns: dic object
    '''
    if (len(fname) == 0):
        return None
    else:
        return loadmat(fname)


class PQMF(nn.Module):
    def __init__(self, N, M, project_root):
        super().__init__()
        self.N = N  # nsubband
        self.M = M  # nfilter
        try:
            assert (N, M) in [(8, 64), (4, 64), (2, 64)]
        except:
            print("Warning:", N, "subbandand ", M, " filter is not supported")

        self.name = str(N) + "_" + str(M) + ".mat"
        self.ana_conv_filter = nn.Conv1d(1, out_channels=N, kernel_size=M, stride=N, bias=False)
        data = load_mat2numpy(op.join(project_root,"models/filters/f_" + self.name))
        data = data['f'].astype(np.float32) / N
        data = np.flipud(data.T).T
        data = np.reshape(data, (N, 1, M)).copy()
        dict_new = self.ana_conv_filter.state_dict().copy()
        dict_new['weight'] = torch.from_numpy(data)
        self.ana_pad = nn.ConstantPad1d((M - N, 0), 0)
        self.ana_conv_filter.load_state_dict(dict_new)

        self.syn_pad = nn.ConstantPad1d((0, M // N - 1), 0)
        self.syn_conv_filter = nn.Conv1d(N, out_channels=N, kernel_size=M // N, stride=1, bias=False)
        gk = load_mat2numpy(op.join(project_root,"models/filters/h_" + self.name))
        gk = gk['h'].astype(np.float32)
        gk = np.transpose(np.reshape(gk, (N, M // N, N)), (1, 0, 2)) * N
        gk = np.transpose(gk[::-1, :, :], (2, 1, 0)).copy()
        dict_new = self.syn_conv_filter.state_dict().copy()
        dict_new['weight'] = torch.from_numpy(gk)
        self.syn_conv_filter.load_state_dict(dict_new)

        for param in self.parameters():
            param.requires_grad = False

    def __analysis_channel(self, inputs):
        return self.ana_conv_filter(self.ana_pad(inputs))

    def __systhesis_channel(self, inputs):
        ret = self.syn_conv_filter(self.syn_pad(inputs)).permute(0, 2, 1)
        return torch.reshape(ret, (ret.shape[0], 1, -1))

    def analysis(self, inputs):
        '''
        :param inputs: [batchsize,channel,raw_wav],value:[0,1]
        :return:
        '''
        ret = None
        for i in range(inputs.size()[1]):  # channels
            if (ret is None):
                ret = self.__analysis_channel(inputs[:, i:i + 1, :])
            else:
                ret = torch.cat((ret,
                                 self.__analysis_channel(inputs[:, i:i + 1, :]))
                                , dim=1)
        return ret

    def synthesis(self, data):
        '''
        :param data: [batchsize,self.N*K,raw_wav_sub],value:[0,1]
        :return:
        '''
        ret = None
        for i in range(data.size()[1]):  # channels
            if (i % self.N == 0):
                if (ret is None):
                    ret = self.__systhesis_channel(data[:, i:i + self.N, :])
                else:
                    new = self.__systhesis_channel(data[:, i:i + self.N, :])
                    ret = torch.cat((ret, new), dim=1)
        return ret

    def forward(self, inputs):
        return self.ana_conv_filter(self.ana_pad(inputs))