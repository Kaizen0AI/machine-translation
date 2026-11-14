## Experiment 1 
## standard gru, 256 hidden size and single layer of encoder-decoder
, ran 10 epochs, early stopping triggered on 10th epochs.
best epoch - 5th → Train Loss: 2.3050, Val Loss: 3.0980, Val Accuracy: 0.4576, LR: 0.001000
last epoch - 10th→ Train Loss: 1.3461, Val Loss: 3.2077, Val Accuracy: 0.4650, LR: 0.000500

Test Corpus BLEU Score: 12.86(after adding detokenization on test set in predict.py score jumped to 13.18)

conclusion- standard case of overfitting
things to imrove - add a dropout layer and increase model capacity(width and length)
aim- to get a bleu score of around 20 without attention model. 

## Experiment 2
## Standard GRU, 512 hidden size and 2 layer with dropout of 0.3
