
import torch.nn as nn


class WritingScoreModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(WritingScoreModel, self).__init__()
        self.linear1 = nn.Linear(input_size, 8)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(8, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        return x


class ReadingScoreModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ReadingScoreModel, self).__init__()
        self.linear1 = nn.Linear(input_size, 8)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(8, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        return x


class MathScoreModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MathScoreModel, self).__init__()
        self.linear1 = nn.Linear(input_size, 8)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(8, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        return x
