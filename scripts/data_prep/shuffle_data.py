import os
import random
import glob
import nltk
from tqdm import tqdm
from transformers import AutoTokenizer

# Ensure nltk data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def get_tokens(text, tokenizer):
    return tokenizer.encode(text, add_special_tokens=False)

def decode_tokens(tokens, tokenizer):
    return tokenizer.decode(tokens)

def shuffle_paragraphs(text):
    paragraphs = text.split('\n\n')
    # Filter empty paragraphs
    paragraphs = [p for p in paragraphs if p.strip()]
    random.shuffle(paragraphs)
    return '\n\n'.join(paragraphs)

def shuffle_sentences_in_paragraph(paragraph):
    try:
        sentences = nltk.sent_tokenize(paragraph)
    
    random.shuffle(sentences)
    return ' '.join(sentences)

def shuffle_sentences(text):
    paragraphs = text.split('\n\n')
    shuffled_paragraphs = []
    for p in paragraphs:
        if not p.strip():
            continue
        shuffled_p = shuffle_sentences_in_paragraph(p)
        shuffled_paragraphs.append(shuffled_p)
    return '\n\n'.join(shuffled_paragraphs)

def shuffle_words_in_sentence(sentence):
    words = sentence.split()
    random.shuffle(words)
    return ' '.join(words)

def shuffle_words(text):
    paragraphs = text.split('\n\n')
    shuffled_paragraphs = []
    for p in paragraphs:
        if not p.strip():
            continue
        sentences = nltk.sent_tokenize(p)
        shuffled_sentences = [shuffle_words_in_sentence(s) for s in sentences]
        shuffled_p = ' '.join(shuffled_sentences)
        shuffled_paragraphs.append(shuffled_p)
    return '\n\n'.join(shuffled_paragraphs)

def main():
    BASE_DIR = "data/medqa"
    OUTPUT_DIRS = ["ordered", "shuffled_para", "shuffled_sent", "shuffled_word"]
    VAL_DIR = os.path.join(BASE_DIR, "val")
    
    # Create directories
    for d in OUTPUT_DIRS:
        os.makedirs(os.path.join(BASE_DIR, d), exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    
    # Initialize tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("allenai/Olmo-3-1025-7B", trust_remote_code=True)
    except:
        print("Could not load Olmo tokenizer, falling back to gpt2 for length calculation")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Load all txt files
    txt_files = glob.glob(os.path.join(BASE_DIR, "textbooks", "*.txt"))
    print(f"Found {len(txt_files)} text files.")
    
    # Process each book individually
    for file_path in tqdm(txt_files, desc="Processing files"):
        filename = os.path.basename(file_path)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            
        # Tokenize to find split point
        tokens = get_tokens(text, tokenizer)
        
        if len(tokens) > 4096:
            val_tokens = tokens[-4096:]
            train_tokens = tokens[:-4096]
        else:
            # If book is too short, take last 10% for val
            split_idx = int(len(tokens) * 0.9)
            val_tokens = tokens[split_idx:]
            train_tokens = tokens[:split_idx]
            
        val_text = decode_tokens(val_tokens, tokenizer)
        train_text = decode_tokens(train_tokens, tokenizer)
        
        # Save Held-out Validation (Per Book)
        with open(os.path.join(VAL_DIR, filename), 'w', encoding='utf-8') as f:
            f.write(val_text)
        
        # generate shuffled versions for this book
        
        # 1. Ordered
        with open(os.path.join(BASE_DIR, "ordered", filename), 'w', encoding='utf-8') as f:
            f.write(train_text)
        
        # 2. Shuffled Paragraphs
        shuffled_para_text = shuffle_paragraphs(train_text)
        with open(os.path.join(BASE_DIR, "shuffled_para", filename), 'w', encoding='utf-8') as f:
            f.write(shuffled_para_text)
        
        # 3. Shuffled Sentences
        shuffled_sent_text = shuffle_sentences(train_text)
        with open(os.path.join(BASE_DIR, "shuffled_sent", filename), 'w', encoding='utf-8') as f:
            f.write(shuffled_sent_text)
            
        # 4. Shuffled Words
        shuffled_word_text = shuffle_words(train_text)
        with open(os.path.join(BASE_DIR, "shuffled_word", filename), 'w', encoding='utf-8') as f:
            f.write(shuffled_word_text)
            
    print("Processing complete.")

if __name__ == "__main__":
    main()
