## Experiment 1 
Baseline model → standard gru, 256 hidden size and single layer of encoder-decoder
, ran 10 epochs, early stopping triggered on 10th epochs.
best epoch - 5th → Train Loss: 2.3050, Val Loss: 3.0980, Val Accuracy: 0.4576, LR: 0.001000
last epoch - 10th→ Train Loss: 1.3461, Val Loss: 3.2077, Val Accuracy: 0.4650, LR: 0.000500

Test Corpus BLEU Score: 12.86(after adding detokenization on test set in predict.py score jumped to 13.18)

conclusion- standard case of overfitting
things to imrove - add a dropout layer and increase model capacity(width and length)
aim - to get a bleu score of around 20 without attention model. 

## Experiment 2
Changes → Deep GRU, 512 hidden size and 2 layer with dropout of 0.3
Aim → Reduce overfitting

earlystopping triggered at 11th epoch
best epoch → 6th →  Train Loss: 2.0927, Val Loss: 2.9389, Val Accuracy: 0.4936, LR: 0.001000
last epoch → 11th → Train Loss: 1.4733, Val Loss: 2.9574, Val Accuracy: 0.5054, LR: 0.000500

Test Corpus BLEU Score: 18.11

conclusion - increasing the model capacity increased the score significantly but its still overfitting, even with dropout(0.3)
next steps - add label smoothing, weight decay and weight tying to the model to improve generalization. 

## Experiment 3
Changes → added label smoothing, weight decay, and weight tying
Aim → Reduce even more overfitting

best epoch → Epoch [12/20], Train Loss: 2.0616, Val Loss: 3.5392, Val Accuracy: 0.5657, LR: 0.000500
last epoch → Epoch [17/20], Train Loss: 1.7910, Val Loss: 3.5754, Val Accuracy: 0.5629, LR: 0.000250

Test Corpus BLEU Score: 0.1

conclusion -  weight tying was a bad idea in this scenario, self.fc and self.embedding sharing the same matrix created an unbalanced loop. label smothing and weight decay are safer for regularization. 
next steps - remove weight tying and increase dropout to 0.4

## Experiment 3a - 
Changes → removed weight tying
Aim → Fix prediction error caused by weight tying the embedding with final layer

best epoch → Epoch [14/20], Train Loss: 2.1258, Val Loss: 3.5368, Val Accuracy: 0.5650, LR: 0.000500
last epoch → Epoch [19/20], Train Loss: 1.8740, Val Loss: 3.5420, Val Accuracy: 0.5682, LR: 0.000250

Test Corpus BLEU Score: 24.19

conclusion - without weight tying this model is at top ends of non-attention RNNs. model is still confusing rare nouns in predictions. 
next step - compare this model to BiLSTM, (expecting only minor improvements).

## Experiment 4 -
Changes → Using BiLSTM instead of GRU
Aim → Compare Standard GRU with BiLSTM before implementing Attention. 

Best epoch → Epoch [20/30], Train Loss: 2.2559, Val Loss: 3.6601, Val Accuracy: 0.5328, LR: 0.000500
Last epoch → Epoch [25/30], Train Loss: 2.0069, Val Loss: 3.6978, Val Accuracy: 0.5347, LR: 0.000250