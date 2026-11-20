import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from predict import load_artifacts

def plot_beam_attention_heatmap(sentence, model, tokenizer, device, beam_width=3, max_length=50, length_penalty=0.7, save_path="images/beam_attention_heatmap.png"):
    model.eval()
    
    with torch.no_grad():
        # 1. Encode source English sentence
        encoded = tokenizer(sentence, add_special_tokens=False)
        src_ids = [tokenizer.bos_token_id] + encoded["input_ids"] + [tokenizer.eos_token_id]
        src_tokens = [tokenizer.decode([tid]).strip() for tid in src_ids]
        
        src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
        attention_mask = torch.ones_like(src_tensor, dtype=torch.long).to(device)
        
        encoder_outputs, encoder_hidden = model.encode(src_tensor)
        
        # 2. Run Beam Search to find the winning token sequence (best_seq)
        beams = [(0.0, [tokenizer.bos_token_id], encoder_hidden)]
        completed_hypotheses = []
        
        for _ in range(max_length):
            all_candidates = []
            
            for score, seq, hidden in beams:
                if seq[-1] == tokenizer.eos_token_id:
                    completed_hypotheses.append((score, seq))
                    continue
                
                current_token = torch.tensor([[seq[-1]]], dtype=torch.long).to(device)
                logits, next_hidden, _ = model.decode_step(
                    current_token, hidden, encoder_outputs, attention_mask
                )
                
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
                top_log_probs, top_ids = torch.topk(log_probs, beam_width)
                
                for i in range(beam_width):
                    next_word_id = top_ids[i].item()
                    new_score = score + top_log_probs[i].item()
                    all_candidates.append((new_score, seq + [next_word_id], next_hidden))
            
            if not all_candidates:
                break
                
            all_candidates.sort(key=lambda x: x[0] / (len(x[1]) ** length_penalty), reverse=True)
            beams = all_candidates[:beam_width]
            
            if len(completed_hypotheses) >= beam_width:
                break
        
        if not completed_hypotheses:
            completed_hypotheses = [(score, seq) for score, seq, _ in beams]
            
        _, best_seq = max(
            completed_hypotheses, 
            key=lambda x: x[0] / (len(x[1]) ** length_penalty)
        )
        
        # 3. Replay best_seq through decode_step to extract clean attention weights
        hidden = encoder_hidden
        pred_tokens = []
        attention_matrix = []
        
        # Iterate up to the second-to-last token (since each step predicts the next token)
        for t in range(len(best_seq) - 1):
            input_token = torch.tensor([[best_seq[t]]], dtype=torch.long).to(device)
            _, hidden, attn_weights = model.decode_step(
                input_token, hidden, encoder_outputs, attention_mask
            )
            
            next_id = best_seq[t + 1]
            if next_id == tokenizer.eos_token_id:
                break
                
            pred_tokens.append(tokenizer.decode([next_id]).strip())
            attention_matrix.append(attn_weights.squeeze(0).cpu().numpy())

    # 4. Plot the clean Beam Search Attention Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(attention_matrix, cmap="viridis")
    fig.colorbar(cax)

    ax.set_xticks(range(len(src_tokens)))
    ax.set_yticks(range(len(pred_tokens)))
    ax.set_xticklabels(src_tokens, rotation=45, ha="left", fontsize=11)
    ax.set_yticklabels(pred_tokens, fontsize=11)

    ax.set_xlabel("English Source Tokens", fontsize=12, labelpad=10)
    ax.set_ylabel("Generated German Tokens (Beam Search)", fontsize=12)
    plt.title("Bahdanau Attention Alignment (Beam Width = 3)", pad=20, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

if __name__ == "__main__":
    model, tokenizer, device = load_artifacts(
        tokenizer_dir="./custom_bpe_tokenizer",
        checkpoint_path="checkpoints/exp5_Attention/best_model.pt",
        hidden_size=512
    )
    
    sample_en = "A Boston Terrier is running on lush green grass in front of a white fence."
    plot_beam_attention_heatmap(sample_en, model, tokenizer, device, beam_width=3)
