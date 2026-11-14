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

        # Tie the final layer to vocab_size as required by CrossEntropyLoss
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