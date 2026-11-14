import sacrebleu
from tqdm import tqdm
from datasets import load_dataset
from predict import load_artifacts, translate_sentence_beam

def evaluate_bleu(num_samples=None):
    # 1. Load your best saved model and BPE tokenizer
    model, tokenizer, device = load_artifacts(
        tokenizer_dir="./custom_bpe_tokenizer",
        checkpoint_path="checkpoints/best_model.pt",
        hidden_size=512
    )
    
    # 2. Load the unseen test split (1,000 sentences in Multi30k)
    dataset = load_dataset("bentrevett/multi30k", split="test")
    if num_samples:
        dataset = dataset.select(range(num_samples))
        
    predictions = []
    references = []
    
    print(f"Generating translations for {len(dataset)} test sentences...")
    for example in tqdm(dataset):
        en_text = example["en"]
        de_ref = example["de"]
        
        # Generate translation using your beam search function
        de_pred = translate_sentence_beam(en_text, model, tokenizer, device, beam_width=3)
        
        predictions.append(de_pred)
        references.append(de_ref)
        
    # 3. Compute SacreBLEU score (expects references as a list of lists)
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    
    print(f"\n==========================================")
    print(f"Test Corpus BLEU Score: {bleu.score:.2f}")
    print(f"==========================================\n")
    
    # Print 3 sample comparisons to inspect qualitative errors
    for i in range(3):
        print(f"Sample {i+1}:")
        print(f"  EN Source : {dataset[i]['en']}")
        print(f"  DE Target : {references[i]}")
        print(f"  DE Pred   : {predictions[i]}\n")

if __name__ == "__main__":
    # Run on all 1,000 test sentences (takes ~30-60 seconds on GPU)
    evaluate_bleu()
