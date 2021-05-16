import argparse
import torch
import random
import numpy as np
import torch.nn.functional as F

from utils import load_data, accuracy
from models import GAT, SpGAT, GCN

# Training settings
parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--dataset', default='cora', help='Enter the dataset')
parser.add_argument('--model', default='GCN', help='Enter the model')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
if args.dataset == 'cora':
    adj, features, labels, idx_train, idx_val, idx_test = load_cora()
else:
    adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset=args.dataset)

model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=int(labels.max()) + 1,
            dropout=args.dropout)


def test_compute():
    model.eval()
    output = model(features, adj)
    a = output[idx_test]
    b = labels[idx_test - min(idx_test)]
    loss_test = F.nll_loss(a, b)
    acc_test = accuracy(output[idx_test], labels[idx_test - min(idx_test)])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data),
          "accuracy= {:.4f}".format(acc_test.data))


def compute_test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(a, b)
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data),
          "accuracy= {:.4f}".format(acc_test.data))


# Restore best model
print('Loading best epoch')
model.load_state_dict(torch.load('{}_{}.pkl'.format(args.dataset, args.model)))

# Testing
if args.dataset == 'cora':
    compute_test()
else:
    test_compute()
