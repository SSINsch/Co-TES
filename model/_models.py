import torch
import torch.nn as nn
import torch.nn.functional as F


def call_bn(bn, x):
    return bn(x)


class NewsNet(nn.Module):
    alg_name = 'FCN'

    def __init__(self, weights_matrix, context_size=1000, hidden_size=300, num_classes=7):
        super(NewsNet, self).__init__()
        n_embed, d_embed = weights_matrix.shape
        self.embedding = nn.Embedding(n_embed, d_embed)
        self.embedding.weight.data.copy_(torch.Tensor(weights_matrix))
        self.avgpool = nn.AdaptiveAvgPool1d(16 * hidden_size)
        self.fc1 = nn.Linear(16 * hidden_size, 4 * hidden_size)
        self.bn1 = nn.BatchNorm1d(4 * hidden_size)
        self.ac = nn.Softsign()
        self.fc2 = nn.Linear(4 * hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        embed = self.embedding(x)  # input (128, 1000)
        embed = embed.detach()  # embed (128, 1000, 300)
        out = embed.view((1, embed.size()[0], -1))  # (1, 128, 300 000)
        out = self.avgpool(out)
        out = out.squeeze(0)
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.ac(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.ac(out)
        out = self.fc3(out)
        # out = F.softmax(out, dim=1)
        return out


class NewsNetCNN(nn.Module):
    alg_name = 'CNN'

    def __init__(self, weights_matrix,
                 input_channel=1, dropout_rate=0.25, momentum=0.1, num_classes=7):
        super(NewsNetCNN, self).__init__()
        self.num_classes = num_classes

        # 임베딩용 변수들
        # n_embed = 단어 개수, d_embed = embedding dimension
        self.n_embed, self.d_embed = weights_matrix.shape
        self.embedding = nn.Embedding(self.n_embed, self.d_embed)
        self.embedding.weight.data.copy_(torch.Tensor(weights_matrix))

        # CNN용 변수들
        kernel_windows = [3, 4, 5]
        self.dropout_rate = dropout_rate
        self.momentum = momentum
        self.input_channel = input_channel
        self.convs = nn.ModuleList(
            [nn.Conv2d(self.input_channel, self.d_embed, kernel_size=(k, self.d_embed)) for k in kernel_windows]
        )

        # 기타 layer
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(len(kernel_windows) * self.d_embed, num_classes)

    def forward(self, x):
        # input (128, 1000) : (128 batch size), (1000 아마 word token 개수??)
        embed = self.embedding(x)
        # => embed (128, 1000, 300) : (128 batch size), (1000 아마 word token 개수??), (300 임베딩 차원)
        embed = embed.unsqueeze(1)
        # => embed (128, 1, 1000, 300)
        conv_x = [conv(embed) for conv in self.convs]
        # => convs_x size = [(128, 300, 998, 1), [(128, 300, 997, 1), [(128, 300, 996, 1)]
        pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2]) for x in conv_x]
        # => pool_x size = [(128, 300, 1), [(128, 300, 1), [(128, 300, 1)]
        linear_x = torch.cat(pool_x, dim=1)
        # => linear_x size = (128, 900, 1)
        linear_x = linear_x.squeeze(-1)
        # => linear_x size = (128, 900)
        linear_drop_x = self.dropout(linear_x)
        logit = self.linear(linear_drop_x)
        # logit = F.softmax(logit, dim=1)

        return logit


class NewsNetLSTM(nn.Module):
    alg_name = 'LSTM'

    def __init__(self, weights_matrix,
                 hidden_size=300, num_classes=7):
        super(NewsNetLSTM, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        # 임베딩용 변수들
        # n_embed = 단어 개수, d_embed = embedding dimension
        self.n_embed, self.d_embed = weights_matrix.shape
        self.embedding = nn.Embedding(self.n_embed, self.d_embed)
        self.embedding.weight.data.copy_(torch.Tensor(weights_matrix))

        # BiLSTM layer 세팅
        self.bi_lstm = nn.LSTM(input_size=self.embedding.embedding_dim,
                               hidden_size=self.hidden_size,
                               batch_first=True,
                               bidirectional=True)

        # bidirectional 이라서 hidden_dim * 2
        self.linear = nn.Linear(self.hidden_size * 2, self.num_classes)

    def forward(self, x):
        # input (128, 1000) : (128 batch size), (1000 아마 word token 개수??)
        embed = self.embedding(x)

        # lstm 통과
        lstm_out, (h_n, c_n) = self.bi_lstm(embed)  # (h_0, c_0) = (0, 0)

        # forward와 backward의 마지막 time-step의 은닉 상태를 가지고 와서 concat
        # 이때 모델이 batch_first라는 점에 주의한다. (dimension 순서가 바뀜)
        hidden = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        out = self.linear(hidden)
        # out = F.softmax(out, dim=1)

        return out
