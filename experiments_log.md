## Experiment 1 
## standard gru, 256 hidden size and single layer of encoder-decoder
, ran 10 epochs, early stopping triggered on 10th epochs.
best epoch - 5th → Train Loss: 2.3050, Val Loss: 3.0980, Val Accuracy: 0.4576, LR: 0.001000
last epoch - 10th→ Train Loss: 1.3461, Val Loss: 3.2077, Val Accuracy: 0.4650, LR: 0.000500

Test Corpus BLEU Score: 12.86(after adding detokenization on test set in predict.py score jumped to 13.18)

conclusion- standard case of overfitting
things to imrove - add a dropout layer and increase model capacity(width and length)
aim - to get a bleu score of around 20 without attention model. 

## Experiment 2
## Standard GRU, 512 hidden size and 2 layer with dropout of 0.3
earlystopping triggered at 11th epoch
best epoch → 6th →  Train Loss: 2.0927, Val Loss: 2.9389, Val Accuracy: 0.4936, LR: 0.001000
last epoch → 11th → Train Loss: 1.4733, Val Loss: 2.9574, Val Accuracy: 0.5054, LR: 0.000500

Test Corpus BLEU Score: 18.11

conclusion - increasing the model capacity increased the score significantly but its still overfitting, even with dropout(0.3)
next steps - add label smoothing, weight decay and weight tying to the model to improve generalization. 