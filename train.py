from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import load_cora
from utils import load_data, accuracy
from models import GAT, SpGAT, GCN, GAT_GCN

# Training settings
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
args.cuda = False
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
if args.dataset == 'cora':
    adj, features, labels, idx_train, idx_val, idx_test = load_cora()
else:
    adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset=args.dataset)

# Model and optimizer
if args.model == 'GAT':
    model = GAT(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=int(labels.max()) + 1,
                dropout=args.dropout,
                nheads=args.nb_heads,
                alpha=args.alpha)
elif args.model == 'GCN':
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=int(labels.max()) + 1,
                dropout=args.dropout)
else:
    model = GAT_GCN(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=int(labels.max()) + 1,
                    dropout=args.dropout,
                    nheads=1,
                    alpha=args.alpha)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

features = Variable(features)
labels = Variable(labels)
adj = Variable(adj)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item()


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


# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    loss_values.append(train(epoch))

    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

torch.save(model.state_dict(), '{}_{}.pkl'.format(args.dataset, args.model))

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}_{}.pkl'.format(args.dataset, args.model)))

# Testing
if args.dataset == 'cora':
    compute_test()
else:
    test_compute()
