import torch
from transformers import PreTrainedTokenizerFast
from model import EncoderDecoderBiLSTM
import torch.nn.functional as F
import re
import os

checkpoint_path = os.getenv("CHECKPOINT_PATH", "checkpoints/best_model.pt")
tokenizer_dir = os.getenv("TOKENIZER_DIR", "custom_bpe_tokenizer")

def clean_detokenize(text): # added after getting a warning from sacrebleu during evaluation
    # Remove spaces before punctuation marks
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    # Fix spaces around hyphens
    text = re.sub(r'\s+-\s+', '-', text)
    return text.strip()


def load_artifacts(tokenizer_dir, checkpoint_path, hidden_size=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the exact BPE vocabulary saved during training
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    vocab_size = len(tokenizer)
    
    # 2. Initialize model architecture and load trained weights
    model = EncoderDecoderBiLSTM(vocab_size, hidden_size, num_layers =2, dropout= 0.4).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle both raw state_dicts and dictionary checkpoints (common with EarlyStopping helpers)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    return model, tokenizer, device

def translate_sentence(sentence, model, tokenizer, device, max_length=50):
    model.eval()
    
    with torch.no_grad():
        # 1. Encode the English input sentence
        encoded = tokenizer(sentence, add_special_tokens=False)
        src_ids = [tokenizer.bos_token_id] + encoded["input_ids"] + [tokenizer.eos_token_id]
        
        # Shape: [1, src_len]
        src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
        
        # 2. Run the Encoder directly (no packing needed for a single sentence)
        src_embedded = model.embedding(src_tensor)
        _, hidden = model.encoder(src_embedded)
        
        # 3. Prepare the initial Decoder input: just [<BOS>]
        # Shape: [1, 1]
        current_token = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).to(device)
        predicted_ids = []
        
        # 4. Autoregressive loop: predict one German token at a time
        for _ in range(max_length):
            trg_embedded = model.embedding(current_token)
            
            # Pass current token and previous hidden state into the decoder
            decoder_out, hidden = model.decoder(trg_embedded, hidden)
            
            # Project to vocabulary logits -> Shape: [1, 1, vocab_size]
            logits = model.fc(decoder_out)
            
            # Pick the highest-scoring token ID
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()
            
            # Stop generating if the model predicts End-of-Sentence
            if next_token_id == tokenizer.eos_token_id:
                break
                
            predicted_ids.append(next_token_id)
            
            # Set the predicted token as the input for the next loop iteration
            current_token = torch.tensor([[next_token_id]], dtype=torch.long).to(device)
            
    # 5. Convert predicted token IDs back into a readable German string
    translation = tokenizer.decode(predicted_ids, skip_special_tokens=True, clean_up_tokenization_spaces = True)
    return translation

def translate_sentence_beam(sentence, model, tokenizer, device, beam_width=3, max_length=50, length_penalty=0.7):
    model.eval()
    
    with torch.no_grad():
        # 1. Encode the English input sentence
        encoded = tokenizer(sentence, add_special_tokens=False)
        src_ids = [tokenizer.bos_token_id] + encoded["input_ids"] + [tokenizer.eos_token_id]
        src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)

        # FIX: Call model.encode() so the bidirectional [4, 1, 512] states pass through 
        # bridge_h and bridge_c to become the (dec_h, dec_c) tuple of shape [2, 1, 512]
        encoder_hidden = model.encode(src_tensor)
        
        # 2. Initialize the beam: list of tuples (cumulative_log_prob, token_id_list, hidden_state)
        beams = [(0.0, [tokenizer.bos_token_id], encoder_hidden)]
        completed_hypotheses = []
        
        for _ in range(max_length):
            all_candidates = []
            
            for score, seq, hidden in beams:
                # If this beam already hit <EOS>, save it and don't expand it further
                if seq[-1] == tokenizer.eos_token_id:
                    completed_hypotheses.append((score, seq))
                    continue
                
                # Feed only the last predicted token into the decoder
                current_token = torch.tensor([[seq[-1]]], dtype=torch.long).to(device)
                trg_embedded = model.embedding(current_token)
                
                decoder_out, next_hidden = model.decoder(trg_embedded, hidden)
                logits = model.fc(decoder_out[:, -1, :])  # Shape: [1, vocab_size]
                
                # Convert logits to log-probabilities so we can add scores safely
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)  # Shape: [vocab_size]
                
                # Grab the top `beam_width` next words for this specific beam
                top_log_probs, top_ids = torch.topk(log_probs, beam_width)
                
                for i in range(beam_width):
                    next_word_id = top_ids[i].item()
                    new_score = score + top_log_probs[i].item()
                    all_candidates.append((new_score, seq + [next_word_id], next_hidden))
            
            # Stop early if all active beams have finished
            if not all_candidates:
                break
                
            # Sort all expanded candidates by length-normalized score and keep top `beam_width`
            all_candidates.sort(key=lambda x: x[0] / (len(x[1]) ** length_penalty), reverse=True)
            beams = all_candidates[:beam_width]
            
            # Stop early if we already collected enough completed sentences
            if len(completed_hypotheses) >= beam_width:
                break
        
        # If no beam hit <EOS> within max_length, fall back to active beams
        if not completed_hypotheses:
            completed_hypotheses = [(score, seq) for score, seq, _ in beams]
            
        # Pick the single best sequence after length normalization
        best_score, best_seq = max(
            completed_hypotheses, 
            key=lambda x: x[0] / (len(x[1]) ** length_penalty)
        )

        # Decode predicted IDs (skipping <BOS> at index 0)
        translation = tokenizer.decode(best_seq[1:], skip_special_tokens=True, clean_up_tokenization_spaces = True)

        cleaned_translation = clean_detokenize(translation)
    
    return cleaned_translation


if __name__ == "__main__":
    model, tokenizer, device = load_artifacts()
    
    test_sentences = [
        "Two young, White males are outside near many bushes.",
        "A man in a blue shirt is standing on a ladder.",
        "A little girl is eating an ice cream cone."
    ]
    
    for en_text in test_sentences:
        de_translation = translate_sentence(en_text, model, tokenizer, device)
        print(f"EN: {en_text}")
        print(f"DE: {de_translation}\n")
