import load_data
import s2s_model
#from masked_cross_entropy import *
from torch import optim
import torch
from torch import nn as nn

import time
import datetime
import math
import random

USE_CUDA = True

MIN_LENGTH = 3
MAX_LENGTH = 25

batch_size = 64

'''
read the whole data set and set up Lang objects (input_lang, output_lang) and sentence pairs
Lang objects contain
  - language name (lang.name)
  - word2index dict (word2index)
  - word counts dict (word2count)
  - index2word dict including  padding, start and end of sequence tokens (index2word)
  - vocab size including padding, start and end of sequence tokens (n_words)
'''

input_lang, output_lang, pairs = load_data.prepare_data('eng', 'deu', MIN_LENGTH, MAX_LENGTH, True)

#removing words with frequencies below MIN_COUNT
MIN_COUNT = 5
input_lang.trim(MIN_COUNT)
output_lang.trim(MIN_COUNT)

print('\nBefore removing infrequent words:')
for i in range(0,11):
    print(pairs[i])

#Remove pairs containing infrequent words (defined by MIN_COUNT)
pairs = load_data.trim_pairs(input_lang, output_lang, pairs)
print('\nAfter removing infrequent words:')
for i in range(0,4):
    print(pairs[i])

'''
Load the data into pyTorch Variables with dimensions [T, B], 
    T is the max sequence length of either the input or output sequence
    B is the batch size, i.e. number of sequence pairs in the batch

E.g. input_var is a sequence of word indices with 0 denoting the padding inde
     input_lengths is a Variable denoting where padding begins, used to truncate the sequence
'''

input_var, input_lengths, target_var, target_lengths = load_data.random_batch(input_lang, output_lang, pairs, batch_size=4)

print('\nSample input batch')
print(input_var)

'''
RUNNING TRAINING
'''
attn_model = 'general'

hidden_size = 256
n_layers = 2
dropout = 0.1
batch_size = 50
input_size = input_lang.n_words
output_size = output_lang.n_words

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 0.5
learning_rate = 0.0001
decoder_learning_ratio = 5.0
#n_epochs = 50000
n_epochs = 10000
epoch = 0
plot_every = 20
print_every = 100
evaluate_every = 100

# Keep track of time elapsed and running averages
start = time.time()
plot_losses = []
print_loss_total = 0 # Reset every print_every
plot_loss_total = 0 # Reset every plot_every

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

# Begin training
ecs = []
dcs = []
eca = 0
dca = 0

model = s2s_model.Model(attn_model, hidden_size, input_size , output_size, n_layers, dropout)
model.cuda()
encoder_optimizer = optim.Adam(model.encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(model.decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

while epoch < n_epochs:
    epoch += 1

    # Get training data for this cycle
    input_batches, input_lengths, target_batches, target_lengths = \
	load_data.random_batch(input_lang, output_lang, pairs,batch_size)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = model( input_batches, input_lengths, target_batches, target_lengths, output_lang, MAX_LENGTH)

    # Clip gradient norms
    ec = torch.nn.utils.clip_grad_norm(model.encoder.parameters(), clip)
    dc = torch.nn.utils.clip_grad_norm(model.decoder.parameters(), clip)

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    # Keep track of loss
    print_loss_total += loss.data[0]
    plot_loss_total += loss.data[0]

    eca += ec
    dca += dc


    if epoch % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print_summary = '%s (%d %d%%) %.4f' % (time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
        print(print_summary)

    if epoch % evaluate_every == 0:
        #Choosing the first sentence in batch to display
        model.training = False
        sentence, attn = model( input_batches[:,:1], input_lengths[:1], target_batches[:,:1], target_lengths[:1], output_lang, MAX_LENGTH)
        model.training = True
        input = input_batches.cpu().data
        output = target_batches.cpu().data

        source_sentence = [input_lang.index2word[input[i,0]] for i in range(input_batches.size(0)) if input[i,0] not in [0,1,2]]
        target_sentence = [output_lang.index2word[output[i,0]] for i in range(target_batches.size(0)) if output[i,0] not in [0,1,2]]

        print(' '.join(source_sentence))
        print(' '.join(target_sentence))
        print(' '.join( sentence))
        print('\n')

