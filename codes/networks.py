import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from .utils import makedirpath

__all__ = ['EncoderHier', 'Encoder', 'PositionClassifier']


class Encoder(nn.Module):
    def __init__(self, K, D=64, bias=True):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 5, 2, 0, bias=bias)
        self.conv2 = nn.Conv2d(64, 64, 5, 2, 0, bias=bias)
        self.conv3 = nn.Conv2d(64, 128, 5, 2, 0, bias=bias)
        self.conv4 = nn.Conv2d(128, D, 5, 1, 0, bias=bias)

        self.K = K
        self.D = D
        self.bias = bias

    def forward(self, x):
        h = self.conv1(x)
        h = F.leaky_relu(h, 0.1)

        h = self.conv2(h)
        h = F.leaky_relu(h, 0.1)

        h = self.conv3(h)

        if self.K == 64:
            h = F.leaky_relu(h, 0.1)
            h = self.conv4(h)

        h = torch.tanh(h)

        return h

    def save(self, name):
        fpath = self.fpath_from_name(name)
        makedirpath(fpath)
        torch.save(self.state_dict(), fpath)

    def load(self, name):
        fpath = self.fpath_from_name(name)
        self.load_state_dict(torch.load(fpath))

    @staticmethod
    def fpath_from_name(name):
        return f'ckpts/{name}/encoder_nohier.pkl'


def forward_hier(x, emb_small, K):
    '''
    Shape: 
        x: (64, 3, 64, 64)   單一個patch
        x{n}: (64, 3, 32, 32) 把64*64的patch分成32*32 4個小patch x1, x2, x3, x4
        xx: (256, 3, 32, 32) 把小patch aggregate成一個single feature p
        hh: (256, 64, 1, 1)  經過兩層conv（128 channel, 64 channel）得出來的output hh 
        h{n}: (64, 64, 1, 1)
        h{nn}: (64, 64, 1, 2)
        h: (64, 64, 2, 2)
    '''
    K_2 = K // 2
    n = x.size(0)    # batch_size: 64
    x1 = x[..., :K_2, :K_2]  # x[..., :32, :32] 第一個dim全取, 第二個dim取32到結束, 第三個dim取0~31
    x2 = x[..., :K_2, K_2:]
    x3 = x[..., K_2:, :K_2]
    x4 = x[..., K_2:, K_2:]

    # print(f"(network.forward_hier) Printing Shape: ")
    # print(f"x shape: {x.shape}")
    # print(f"x.size(0) n: {n}")
    # print(f"x1 shape: {x1.shape}")
    # print(f"x2 shape: {x2.shape}")
    # print(f"x3 shape: {x3.shape}")
    # print(f"x4 shape: {x4.shape}")

    xx = torch.cat([x1, x2, x3, x4], dim=0)   # feature aggregate 成 feature p (aggregate to produce a single feature)
    # print(f"xx shape: {xx.shape}") 

    hh = emb_small(xx)
    # print(f"hh shape: {hh.shape}")

    h1 = hh[:n]
    h2 = hh[n: 2 * n]
    h3 = hh[2 * n: 3 * n]
    h4 = hh[3 * n:]
    # print(f"h1 shape: {h1.shape}")
    # print(f"h2 shape: {h2.shape}")
    # print(f"h3 shape: {h3.shape}")
    # print(f"h4 shape: {h4.shape}")

    h12 = torch.cat([h1, h2], dim=3)
    h34 = torch.cat([h3, h4], dim=3)
    h = torch.cat([h12, h34], dim=2)
    # print(f"h12 shape: {h12.shape}")
    # print(f"h34 shape: {h34.shape}")
    # print(f"h shape: {h.shape}")
    
    return h


class EncoderDeep(nn.Module):
    def __init__(self, K, D=64, bias=True):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=bias)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 0, bias=bias)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 0, bias=bias)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 0, bias=bias)
        self.conv5 = nn.Conv2d(128, 64, 3, 1, 0, bias=bias)
        self.conv6 = nn.Conv2d(64, 32, 3, 1, 0, bias=bias)
        self.conv7 = nn.Conv2d(32, 32, 3, 1, 0, bias=bias)
        self.conv8 = nn.Conv2d(32, D, 3, 1, 0, bias=bias)

        self.K = K
        self.D = D

    def forward(self, x):
        h = self.conv1(x)
        h = F.leaky_relu(h, 0.1)

        h = self.conv2(h)
        h = F.leaky_relu(h, 0.1)

        h = self.conv3(h)
        h = F.leaky_relu(h, 0.1)

        h = self.conv4(h)
        h = F.leaky_relu(h, 0.1)

        h = self.conv5(h)
        h = F.leaky_relu(h, 0.1)

        h = self.conv6(h)
        h = F.leaky_relu(h, 0.1)

        h = self.conv7(h)
        h = F.leaky_relu(h, 0.1)

        h = self.conv8(h)
        h = torch.tanh(h)

        return h

    def save(self, name):
        fpath = self.fpath_from_name(name)
        makedirpath(fpath)
        torch.save(self.state_dict(), fpath)

    def load(self, name):
        fpath = self.fpath_from_name(name)
        self.load_state_dict(torch.load(fpath))

    @staticmethod
    def fpath_from_name(name):
        return f'ckpts/{name}/encdeep.pkl'


class EncoderHier(nn.Module):
    def __init__(self, K, D=64, bias=True):
        super().__init__()

        '''
        因為瑕疵大小不確定, 用多尺度的encoder可以幫忙識別各種大小的瑕疵。
        文章採用 K=64, 32 這兩種大小

        將(64, 64)的patch分成4個(32, 32)的小patch
        f_small：  
            input (1, 32, 32)
            output(64, 1, 1)
        f_big:
            input (batchsize, 64, 2, 2)
            中間包含兩個convolution, 分別是 128 channel的2*2 和 64 channel的1*1, 
            將輸出轉換成 output (batchsize, 64, 1, 1)
        '''

        if K > 64:   # K 什麼時候會大於 64
            self.enc = EncoderHier(K // 2, D, bias=bias)

        elif K == 64:
            self.enc = EncoderDeep(K // 2, D, bias=bias)  # D 是embedding output的dimension, K是receptive field（patch size）

        else:
            raise ValueError()

        # f_big 中間的兩個convolution, 分別是 128 channel 和 64
        self.conv1 = nn.Conv2d(D, 128, 2, 1, 0, bias=bias)
        self.conv2 = nn.Conv2d(128, D, 1, 1, 0, bias=bias)

        self.K = K
        self.D = D

    def forward(self, x):
        h = forward_hier(x, self.enc, K=self.K)   # patch, encoder_small(f_small), patchsize

        h = self.conv1(h)
        h = F.leaky_relu(h, 0.1)

        h = self.conv2(h)
        h = torch.tanh(h)

        return h

    def save(self, name):
        fpath = self.fpath_from_name(name)
        makedirpath(fpath)
        torch.save(self.state_dict(), fpath)

    def load(self, name):
        fpath = self.fpath_from_name(name)
        self.load_state_dict(torch.load(fpath))

    @staticmethod
    def fpath_from_name(name):
        return f'ckpts/{name}/enchier.pkl'


################


xent = nn.CrossEntropyLoss()


class NormalizedLinear(nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(NormalizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        with torch.no_grad():
            w = self.weight / self.weight.data.norm(keepdim=True, dim=0)
        return F.linear(x, w, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class PositionClassifier(nn.Module):
    def __init__(self, K, D, class_num=8):
        super().__init__()
        self.D = D

        self.fc1 = nn.Linear(D, 128)
        self.act1 = nn.LeakyReLU(0.1)

        self.fc2 = nn.Linear(128, 128)
        self.act2 = nn.LeakyReLU(0.1)

        self.fc3 = NormalizedLinear(128, class_num)

        self.K = K

    def save(self, name):
        fpath = self.fpath_from_name(name)
        makedirpath(fpath)
        torch.save(self.state_dict(), fpath)

    def load(self, name):
        fpath = self.fpath_from_name(name)
        self.load_state_dict(torch.load(fpath))

    def fpath_from_name(self, name):
        return f'ckpts/{name}/position_classifier_K{self.K}.pkl'

    @staticmethod
    def infer(c, enc, batch):
        '''
        c: classifier
        enc: encoder (EncoderHier)
        '''

        x1s, x2s, ys = batch

        h1 = enc(x1s)
        h2 = enc(x2s)

        logits = c(h1, h2)
        loss = xent(logits, ys)
        return loss

    def forward(self, h1, h2):
        h1 = h1.view(-1, self.D)
        h2 = h2.view(-1, self.D)

        h = h1 - h2

        h = self.fc1(h)
        h = self.act1(h)

        h = self.fc2(h)
        h = self.act2(h)

        h = self.fc3(h)
        return h

