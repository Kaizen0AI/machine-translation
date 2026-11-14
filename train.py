from data import get_dataloaders
import torch
import torch.nn as nn
import torch.optim as optim
from model import EncoderDecoderGRU
from tqdm import tqdm
import os
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.helper_func import EarlyStoppingAndCheckpoint

def train_model(device, model, criterion, optimizer, train_loader, val_loader, scheduler, tracker, pad_id, num_epochs= 10):

    def validate(model, val_loader, criterion, device):
        model.eval()
        running_val_loss = 0.0
        correct_predictions = 0
        total_real_tokens = 0

        with torch.no_grad(): # disables gradient computation for validation, saving memory and computation time
            for val_batch in val_loader:
                val_input_ids = val_batch["input_ids"].to(device)
                val_input_lengths = val_batch["input_lengths"].to(device)
                val_labels = val_batch["labels"].to(device)

                # shift targets: input has [<BOS>, w1, w2], target has [w1, w2, <EOS>], 
                # so we need to shift the target labels by one position to the right for teacher forcing
                decoder_input = val_labels[:, :-1]  # all tokens except the last one
                target_labels = val_labels[:, 1:]  # all tokens except the first one

                # Forward pass
                val_outputs = model(val_input_ids, val_input_lengths, decoder_input)

                # Compute loss/ cross-entropy expects (N, C) for logits and (N) for labels,
                # so we flatten the outputs and labels(N = batch_size * seq_len = total_tokens, C = vocab_size)
                loss = criterion(val_outputs.reshape(-1, val_outputs.size(-1)), target_labels.reshape(-1)) 
                running_val_loss += loss.item() # adds up the loss for this batch to the running total for the epoch

                # Calculate accuracy ONLY on non-padding tokens
                _, predicted = torch.max(val_outputs, dim=-1)
                non_pad_mask = (target_labels != pad_id)  # mask to ignore padding tokens
                correct_predictions += ((predicted == target_labels) & non_pad_mask).sum().item() #.item() converts the tensor to a Python number
                total_real_tokens += non_pad_mask.sum().item() # counts non-padded tokens for accuracy calculation

        val_loss = running_val_loss / len(val_loader)
        val_accuracy = correct_predictions / total_real_tokens if total_real_tokens > 0 else 0

        return val_loss, val_accuracy

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epochs [{epoch + 1}/{num_epochs}]", leave= True)

        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            labels = batch["labels"].to(device)

            # shift targets for teacher forcing
            decoder_input = labels[:, :-1]  # all tokens except the last one
            target_labels = labels[:, 1:]  # all tokens except the first one

            optimizer.zero_grad() #resets gradients before every batch to avoid accumulation from previous batches

            # Forward pass
            outputs = model(input_ids, input_lengths, decoder_input)

            # Compute loss (using .reshape() safely handles non-contiguous sliced tensors)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), target_labels.reshape(-1))
            running_train_loss += loss.item() # adds up the loss for this batch to the running total for the epoch

            # Backward pass, gradient clipping and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loop.set_postfix({"train_loss": f"{loss.item():.4f}"})

        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        epoch_train_loss = running_train_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, LR: {current_lr:.6f}")

        scheduler.step(val_loss)  # Step the scheduler based on validation loss

        tracker(val_loss, model, optimizer, epoch)  # Check for early stopping and save checkpoint if needed

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        history_path = os.getenv("HISTORY_PATH", "checkpoints/training_history.json")
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        with open(history_path, "w") as f:
            json.dump(history, f)

        if tracker.early_stop:
            print("Early stopping triggered. Stopping training.")
            break

    return history

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, tokenizer = get_dataloaders()

    # using len(tokenizer) instead of tokenizer.vocab_size to ensure we get the correct size of the vocabulary, including special tokens.
    vocab_size = len(tokenizer)
    hidden_size = 256  # we can adjust this based on your model's complexity and available resources

    model = EncoderDecoderGRU(vocab_size, hidden_size).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    tracker = EarlyStoppingAndCheckpoint(patience=5, min_delta=1e-4)

    history = train_model(device=device,
                          model=model,
                          criterion=criterion,
                          optimizer=optimizer,
                          train_loader=train_loader,
                          val_loader=val_loader,
                          scheduler=scheduler,
                          tracker=tracker,
                          pad_id=tokenizer.pad_token_id,
                          num_epochs=10)