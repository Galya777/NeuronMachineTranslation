import pandas as pd
import re
import os

def preprocess_transliteration(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Remove modern scribal notations
    # ! (certain), ? (questionable), / (line divider), : or . (word divider)
    text = re.sub(r'[!?/:\.\(\)˹˺]', '', text)
    
    # < > (scribal insertions, but keep text inside)
    # text = re.sub(r'<([^>]*)>', r'\1', text) # Note: keep text inside or remove entirely?
    # Instruction says: < > (scribal insertions, but keep the text in translit / translations)
    text = text.replace('<', '').replace('>', '')
    
    # [ ] (broken signs/lines, remove brackets but keep text)
    text = text.replace('[', '').replace(']', '')
    
    # 2. Replace breaks, gaps, superscripts, subscripts
    # [x] -> <gap>
    text = text.replace('x', '<gap>')
    # ... or [... ...] -> <big_gap>
    text = text.replace('...', '<big_gap>')
    
    # 3. Specific character substitutions
    # Ḫ ḫ --> H h
    text = text.replace('ḫ', 'h').replace('Ḫ', 'H')
    
    # Unicode chars to simple chars if needed, but the model can handle them if they are in vocab.
    # Instruction says: á -> a2, etc. (standard CDLI/ORACC)
    
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_translation(text):
    if not isinstance(text, str):
        return ""
    # Standard cleaning for English translation
    text = text.replace('\n', ' ')
    # Remove some common OCR/parsing noise if any
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def prepare_files(input_csv):
    df = pd.read_csv(input_csv)
    
    # Shuffle for train/dev split
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 90% train, 10% dev
    split_idx = int(0.9 * len(df))
    train_df = df.iloc[:split_idx]
    dev_df = df.iloc[split_idx:]
    
    os.makedirs('akkadian_data', exist_ok=True)
    
    with open('akkadian_data/train.akk', 'w', encoding='utf-8') as f_akk, \
         open('akkadian_data/train.en', 'w', encoding='utf-8') as f_en:
        for _, row in train_df.iterrows():
            akk = preprocess_transliteration(row['transliteration'])
            en = preprocess_translation(row['translation'])
            if akk and en:
                f_akk.write(akk + '\n')
                f_en.write(en + '\n')

    with open('akkadian_data/dev.akk', 'w', encoding='utf-8') as f_akk, \
         open('akkadian_data/dev.en', 'w', encoding='utf-8') as f_en:
        for _, row in dev_df.iterrows():
            akk = preprocess_transliteration(row['transliteration'])
            en = preprocess_translation(row['translation'])
            if akk and en:
                f_akk.write(akk + '\n')
                f_en.write(en + '\n')

if __name__ == "__main__":
    train_csv = 'Akkadian to English/deep-past-initiative-machine-translation/train.csv'
    prepare_files(train_csv)
    print("Preprocessing finished. Files saved in akkadian_data/")
