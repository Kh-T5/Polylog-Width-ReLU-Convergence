import torch
import torch.nn as nn
import torch.nn.functional as F


class ShallowNet(nn.Module):
    """
    Class for a shallow neural network with scalar output, initialized and scaled as assumed in Ji & Telgarsky paper.
    """

    def __init__(self, d, m, scale_output=True):
        """
        Inputs:
            m, int, represents hidden_size
            d, int, represents data feature size
            scale_output, bool, scaling for model output corresponding to paper's scaling.
        """
        super().__init__()
        self.m = m
        self.hidden = nn.Linear(d, m, bias=False)
        self.a = nn.Parameter(torch.randn(m, 1), requires_grad=False)

        nn.init.normal_(self.hidden.weight, std=1.0)

        self.scaling = 1.0 / torch.sqrt(torch.tensor(m)) if scale_output else 1.0

    def forward(self, X):
        """
        Computes forward pass in network.
        Input:
            - X, (batchsize, d)
        """
        h = F.relu(self.hidden(X))
        return (h @ self.a) * self.scaling

    def get_activations(self, X):
        """
        Tracks activations in hidden layer given input X: (batchsize, d)
        """
        return (self.hidden(X) > 0).float()


class Teacher(nn.Module):
    """
    Teacher class in the experiment, used as ground truth for labels.
    It allows us to isolate the effect of Width (m) from other variables like data noise or label corruption.
    If the Student fails, we know it is because of its own architectural or optimization constraints, not the data quality.
    """

    def __init__(self, d, k):
        """
        Initialize weights with normal distribution.
        """
        super().__init__()
        self.hidden = nn.Linear(d, k, bias=False)
        nn.init.normal_(self.hidden.weight, std=1.0)
        self.a = torch.randn(k, 1).sign()

    @torch.no_grad()
    def forward(self, x):
        # f*(x) = sum(a_i * relu(w_i * x))
        h = F.relu(self.hidden(x))
        return h @ self.a


def logistic_loss(logits, targets):
    """
    Logistic loss for binary classification.
    """
    loss = torch.mean(torch.log(1 + torch.exp(-targets * logits)))
    return loss
