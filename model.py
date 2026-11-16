import torch.nn as nn
import torch

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

    def forward(self, src, src_lengths, trg):
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
        decoder_out = self.decoder(trg_embedded, (dec_h, dec_c))

        return self.fc(decoder_out)
    