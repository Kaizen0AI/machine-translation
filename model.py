import torch.nn as nn
import torch
import torch.nn.functional as F

class EncoderDecoderGRU(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, dropout):
        super(EncoderDecoderGRU, self).__init__()

        # using the shared vocabulary size from our custom BPE tokenizer
        self.embedding = nn.Embedding(vocab_size, hidden_size) 

        self.dropout = nn.Dropout(dropout)

        # setting input_size=embedding_size=hidden_size for seemless integration with the GRU layers(smooth encoder-decoder handshake)
        self.encoder = nn.GRU(hidden_size, hidden_size,
                              num_layers = num_layers,
                              dropout=dropout if num_layers > 1 else 0,
                              batch_first=True)
        
        self.decoder = nn.GRU(hidden_size, hidden_size,
                              num_layers=num_layers,
                              dropout=dropout if num_layers > 1 else 0,
                              batch_first=True)

        self.fc = nn.Linear(hidden_size, vocab_size)

    def encode(self, src, src_lengths=None):
        src_embedded = self.dropout(self.embedding(src))
        if src_lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                src_embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, h_n = self.encoder(packed)
        else:
            _, h_n = self.encoder(src_embedded)
        return h_n


    def forward(self, src, src_lengths, trg):
        '''
        # - ENCODER STEP -
        # Embed the English source text
        src_embedded = self.dropout(self.embedding(src))

        # Pack the sequence so the GRU ingnores the padded tokens
        packed_embedding = nn.utils.rnn.pack_padded_sequence(
            src_embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False

        ) # enforce_sorted=false is vital because our dataloader delivers random batch orderings.
          # it sorts the random sized sentences for GPU to process efficiently and un-sort them back to their original order after processing.
          # randonness is important for training stability and generalization, so we don't want to sort the batches before passing them to the model.
          # if not sorted, the GRU will throw an error because it expects sorted sequences when using packed sequences.

        # pass the packed seq through the encoder, we only need the final hidden state(h_n) for the decoder
        _, h_n = self.encoder(packed_embedding)
        '''
        h_n = self.encode(src, src_lengths)
        # - DECODER STEP -
        # Embed the German target text (Teacher forcing)
        trg_embedding = self.dropout(self.embedding(trg))

        # pass target embeddings through decoder, using encoders h_n as the context bridge
        decoder_out, _ = self.decoder(trg_embedding, h_n)

        # project the hidden representations to vocabulary logits 
        logits = self.fc(decoder_out)

        return logits

class EncoderDecoderBiLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Bidirectional encoder
        self.encoder = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            dropout= dropout if num_layers > 1 else 0,
            bidirectional = True,
            batch_first=True
        )

        # Bridge layers to compress [forward; backward] (2 * hidden_size) -> hidden_size
        self.bridge_h = nn.Linear(hidden_size * 2, hidden_size)
        self.bridge_c = nn.Linear(hidden_size * 2, hidden_size)

        # Unidirectional Decoder
        self.decoder = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def encode(self, src, src_lengths=None):
        src_embedded = self.dropout(self.embedding(src))

        if src_lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                src_embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h_n, c_n) = self.encoder(packed)
        else:
            # Used during single-sentence inference in predict.py
            _, (h_n, c_n) = self.encoder(src_embedded)

        # h_n & c_n shapes: [num_layers * 2, batch_size, hidden_size]
        # Reshape to separate layers from directions: [num_layers, 2, batch_size, hidden_size]
        batch_size = src.size(0)
        h_n = h_n.view(self.num_layers, 2, batch_size, self.hidden_size)
        c_n = c_n.view(self.num_layers, 2, batch_size, self.hidden_size)

        # Concatenate forward (index 0) and backward (index 1) along feature dim -> [num_layers, batch, hidden * 2]
        h_cat = torch.cat((h_n[:, 0, : ,:], h_n[:, 1, :, :]), dim=-1)
        c_cat = torch.cat((c_n[:, 0, :, :], c_n[:, 1, :, :]), dim=-1)

        # Compress through bridge layers with tanh activation -> [num_layers, batch_size, hidden]
        dec_h = torch.tanh(self.bridge_h(h_cat))
        dec_c = torch.tanh(self.bridge_c(c_cat))

        return dec_h, dec_c

    def forward(self, src, src_lengths, trg):
        dec_h, dec_c = self.encode(src, src_lengths)

        trg_embedded = self.dropout(self.embedding(trg))
        decoder_out, _ = self.decoder(trg_embedded, (dec_h, dec_c))

        return self.fc(decoder_out)

class BahdanauAttention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super().__init__()
        # Scores alignment between decoder hidden state and all encoder outputs
        self.attn = nn.Linear(enc_hidden_dim + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)

    def forward(self, dec_hidden, encoder_outputs, attention_mask):
        # dec_hidden: [batch_size, dec_hidden_dim] (top layer of decoder)
        # encoder_outputs: [batch_size, src_len, enc_hidden_dim]
        # attention_mask: [batch_size, src_len] (1 for real tokens, 0 for <PAD>)

        src_len = encoder_outputs.size(1)

        # Repeat decoder hidden state across all source tine steps -> [batch, src_len, dec_hidden_dim]
        hidden_expanded = dec_hidden.unsqueeze(1).repeat(1, src_len, 1) 
        # unsqueeze(1) adds a dummy dimension of size 1 at index 1, repeat(1, src_len, 1),
        # repeats(copies) the tensor along all 3 dims the specified number of times,
        # this converts [batch, dec_hidden_dim] tensor to [batch, src_len, dec_hidden_dim], so it can be concatanated with encoder_outputs tensor of same dims

        # Compute alignment energy scores -> [batch, src_len, dec_hidden_dim]
        energy = torch.tanh(self.attn(torch.cat((hidden_expanded, encoder_outputs), dim=-1))) # a tanh layer to learn "e" for the attn weigth computaion

        # Project to a single scalar score per English word -> [batch, src_len]
        scores = self.v(energy).squeeze(-1)

        # Mask out <PAD> positions so softmax gives them 0.0 weight
        scores = scores.masked_fill(attention_mask == 0, -1e9) # softmax of -infintity(-1e9) is zero

        # Convert to attention probabilities summing to 1.0 -> [batch, src_len]
        attn_weights = F.softmax(scores, dim=-1)

        # Compute weighted sum of encoder outputs -> [batch, 1, enc_hidden_dim]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        # bmm = batch matrix multiplication; [batch, (1, src_len)] * [batch, (src_len, enc_hidden_dim)] = [batch, (1, enc_hidden_dim)]
        return context, attn_weights

class EncoderDecoderAttention(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        enc_dim = hidden_size * 2 # bidirectional encoder outputs 1024 features

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.encoder = nn.LSTM(
            hidden_size, hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional = True,
            batch_first=True
        )

        # Pure linear bridges (preserves unbounded LSTM cell state scale)
        self.bridge_h = nn.Linear(enc_dim, hidden_size)
        self.bridge_c = nn.Linear(enc_dim, hidden_size)

        self.attention = BahdanauAttention(enc_dim, hidden_size)

        # decoder input combines target word embedding + attention context vector
        self.decoder = nn.LSTM(
            hidden_size + enc_dim, hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional = False,
            batch_first=True 
        )

        # final classifier combines decoder input + attention context + embedding
        self.fc = nn.Linear(hidden_size + enc_dim + hidden_size, vocab_size)

    def encode(self, src, src_lengths = None):
        src_embedded = self.dropout(self.embedding(src))

        if src_lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                src_embedded, src_lengths.cpu(), batch_first= True, enforce_sorted = False
            )
            packed_outputs, (h_n, c_n) = self.encoder(packed)
            # unpack all time steps back to padded tensor -> [batch, src_len< hidden_size *2]
            encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first= True)
        else: 
            encoder_outputs, (h_n, c_n) = self.encoder(src_embedded)

        batch_size = src.size(0)
        h_n = h_n.view(self.num_layers, 2, batch_size, self.hidden_size)
        c_n = c_n.view(self.num_layers, 2, batch_size, self.hidden_size)

        h_cat = torch.cat((h_n[:, 0, :, :], h_n[:, 1, :, :]), dim=-1)
        c_cat = torch.cat((c_n[:, 0, :, :], c_n[:, 1, :, :]), dim=-1)

        dec_h = self.bridge_h(h_cat)
        dec_c = self.bridge_c(c_cat)

        return encoder_outputs, (dec_h, dec_c)

    def decode_step(self, input_token, hidden_state, encoder_outputs, attention_mask):
        # input_token: [batch_size,1]
        embedded = self.dropout(self.embedding(input_token)) # [batch, 1, hidden_size]

        # use top deocder layer's hidden state to query attention
        top_dec_hidden = hidden_state[0][-1] # [batch, hidden_size]
        context, attn_weights = self.attention(top_dec_hidden, encoder_outputs, attention_mask)

        # concatenate word embedding and attention context -> [batch, 1, hidden_size + enc_dim]
        rnn_input = torch.cat((embedded, context), dim=-1)

        decoder_out, next_hidden = self.decoder(rnn_input, hidden_state)

        # combine deocder output, context, and embedding for rich vocabulary prediction
        combined = torch.cat((decoder_out, context, embedded), dim=-1).squeeze(1)
        logits = self.fc(combined) # [batch, vocab_size]

        return logits, next_hidden, attn_weights

    def forward(self, src, src_lengths, trg, attention_mask):
        encoder_outputs, hidden_state = self.encode(src, src_lengths)

        batch_size, trg_len = trg.size()
        vocab_size = self.fc.out_features

        outputs = torch.zeros(batch_size, trg_len, vocab_size, device= src.device)

        for t in range(trg_len):
            input_token = trg[:, t].unsqueeze(1)
            logits, hidden_state, _ = self.decode_step(
                input_token, hidden_state, encoder_outputs, attention_mask 
            )
            outputs[:, t, :] = logits

        return outputs


    