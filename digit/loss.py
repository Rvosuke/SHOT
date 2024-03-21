import torch
import torch.nn as nn


def entropy(input_):
    bs = input_.size(0)
    entropy_ = -input_ * torch.log(input_ + 1e-5)
    entropy_ = torch.sum(entropy_, dim=1)
    return entropy_


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, size_average=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.size_average = size_average
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.log_softmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu:
            targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        if self.size_average:
            loss = (- targets * log_probs).mean(0).sum()
        else:
            loss = (- targets * log_probs).sum(1)
        return loss
