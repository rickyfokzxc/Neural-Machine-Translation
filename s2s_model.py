import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from masked_cross_entropy import *

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        # input_seqs = [TxB]
        embedded = self.embedding(input_seqs) #embedded = [TxBxD]
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths) #GRU ignores padded elements
        outputs, hidden = self.gru(packed, hidden) #If hidden == None, sets to zero
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs) # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs [2*hidden_size]
        return outputs, hidden

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs,USE_CUDA=True):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(this_batch_size, max_len)) # B x T'

        if USE_CUDA:
            attn_energies = attn_energies.cuda()

        # For each batch of encoder outputs, vectorize this
        # Given hidden [1,B,D] (one time step in decoder)
        # encoder_outputs [S,B,D]
        # Dot over D
        if self.method == 'general':
            encoder_outputs = self.attn(encoder_outputs)

        hidden = hidden.expand(max_len,-1,-1).contiguous()
        if self.method in ['dot','general']:
            hidden = hidden.view(-1,self.hidden_size)
            attn_energies = hidden.mm(encoder_outputs.view(-1,self.hidden_size).t().contiguous()).diag()
            attn_energies = attn_energies.view(max_len,this_batch_size).t().contiguous()
        elif self.method == 'concat':
            encoder_outputs = self.attn(torch.cat((hidden, encoder_outputs), 2)).view(-1,hidden_size)
            attn_energies = torch.matmul(encoder_outputs, self.v.t())
            attn_energies = attn_energies.view(max_len,this_batch_size).t().contiguous()

        # Normalize energies to weights in range 0 to 1, resize to B x 1 x T' 
        return F.softmax(attn_energies).unsqueeze(1)

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs):
        # Note: we run this one step at a time
        # input_seq = [B], last_hidden = [num_layers x B x D]
        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq) #[B,D]
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.hidden_size) # 1 x B x D

        # Get current hidden state from input word and last hidden state
        # rnn_output = [1xBxD], hidden = [num_layers x B x D]
        rnn_output, hidden = self.gru(embedded, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs) #[ Bx1xT]
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # [Bx1xT] * BxTxD = Bx1xD

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0) # S=1 x B x D -> B x D
        context = context.squeeze(1)       # B x S=1 x D -> B x D
        concat_input = torch.cat((rnn_output, context), 1) #concat layer
        concat_output = F.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights

class Model(nn.Module):
    def __init__(self,attn_model, hidden_size, input_size, output_size, n_layers, dropout): #create encoder-decoder weights here
        super().__init__()
        self.encoder = EncoderRNN(input_size, hidden_size, n_layers, dropout)
        self.decoder = LuongAttnDecoderRNN(attn_model, hidden_size, output_size, n_layers, dropout)

    def forward(self, input_batches, input_lengths, target_batches, target_lengths, output_lang, max_length,
                SOS_token=1, EOS_token=2):
        #encoder outputs
        encoder_outputs, encoder_hidden = self.encoder(input_batches, input_lengths, None)
        
        #Setting up decoder things
        max_target_length = max(target_lengths)
        batch_size = input_batches.size(1)
        decoder_hidden = encoder_hidden[:self.decoder.n_layers] # Use last (forward) hidden state from encoder
        decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size)).cuda()
        all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, self.decoder.output_size)).cuda()

        #Decoder in training mode
        if self.training == 1: #decoder in training mode
            for t in range(max_target_length):
                decoder_output, decoder_hidden, decoder_attn = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                all_decoder_outputs[t] = decoder_output
                decoder_input = target_batches[t] # Teacher forcing, Next input is current target
            loss = masked_cross_entropy(
                all_decoder_outputs.transpose(0, 1).contiguous(), #  B x T' x vocab
                target_batches.transpose(0, 1).contiguous(), #  B x T
                target_lengths
            )
            return loss
        else: #Evaluation mode, only for B=1
            assert input_batches.size(1) == 1, "Evaluation mode only for batch_size==1"
            self.encoder.train(False)
            self.decoder.train(False)
            #input_batches.volatile = True
            #decoder_input.volatile = True
            decoded_words = []
            target_words = []
            decoder_attentions = torch.zeros(max_length + 1, max_length + 1)

            for di in range(max_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                #attention [B=1,1,T]
                decoder_attentions[di,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

                # Choose top word from output
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]
                if ni == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(output_lang.index2word[ni])
            
        
                # Next input is chosen word
                decoder_input = Variable(torch.LongTensor([ni])).cuda()
    
            self.encoder.train(True)
            self.decoder.train(True)
            return decoded_words, decoder_attentions[:di+1, :len(encoder_outputs)]
