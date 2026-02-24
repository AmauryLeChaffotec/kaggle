---
name: kaggle-nlp
description: Expert en compétitions Kaggle de NLP (Natural Language Processing). Utiliser quand l'utilisateur travaille sur une compétition avec du texte (classification, NER, question answering, summarization).
argument-hint: <type_tâche_nlp ou stratégie>
---

# Expert NLP - Kaggle Gold Medal

Tu es un expert en compétitions Kaggle NLP. Tu maîtrises les Transformers, le fine-tuning, et les stratégies modernes de NLP.

## Modèles Recommandés (2024-2025)

### Classification de Texte
| Modèle | Taille | Performance | Recommandé pour |
|---|---|---|---|
| DeBERTa-v3-large | 435M | Très haute | SOTA classification |
| DeBERTa-v3-base | 86M | Haute | Bon compromis |
| RoBERTa-large | 355M | Haute | Robuste |
| Llama-3.1-8B (LoRA) | 8B | Très haute | Si GPU suffisant |
| Mistral-7B (LoRA) | 7B | Très haute | Efficace |

### NER (Named Entity Recognition)
- DeBERTa-v3-large + token classification head
- SpanMarker

### Question Answering
- DeBERTa-v3-large + QA head
- Longformer pour documents longs

## Pipeline NLP Complet

### 1. Configuration

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
)
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from tqdm import tqdm

class CFG:
    seed = 42
    model_name = 'microsoft/deberta-v3-large'
    max_length = 512
    batch_size = 8
    epochs = 5
    lr = 2e-5
    weight_decay = 0.01
    warmup_ratio = 0.1
    n_folds = 5
    num_classes = 2
    gradient_accumulation_steps = 2
    max_grad_norm = 1.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### 2. Tokenization et Dataset

```python
class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, target_col=None):
        self.texts = df['text'].values  # ADAPTER
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.targets = df[target_col].values if target_col else None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
        }

        if 'token_type_ids' in encoding:
            item['token_type_ids'] = encoding['token_type_ids'].squeeze()

        if self.targets is not None:
            item['labels'] = torch.tensor(self.targets[idx], dtype=torch.long)

        return item

# Pour les paires de textes (NLI, similarity, etc.)
class TextPairDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, target_col=None):
        self.text1 = df['text1'].values
        self.text2 = df['text2'].values
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.targets = df[target_col].values if target_col else None

    def __len__(self):
        return len(self.text1)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.text1[idx]),
            str(self.text2[idx]),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        item = {k: v.squeeze() for k, v in encoding.items()}
        if self.targets is not None:
            item['labels'] = torch.tensor(self.targets[idx], dtype=torch.long)
        return item
```

### 3. Modèle Custom avec Pooling Avancé

```python
class NLPModel(nn.Module):
    def __init__(self, model_name, num_classes, pooling='mean'):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        hidden_size = self.backbone.config.hidden_size

        if pooling == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.Tanh(),
                nn.Linear(256, 1)
            )

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if token_type_ids is not None:
            kwargs['token_type_ids'] = token_type_ids

        outputs = self.backbone(**kwargs)
        last_hidden = outputs.last_hidden_state

        if self.pooling == 'cls':
            pooled = last_hidden[:, 0]
        elif self.pooling == 'mean':
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (last_hidden * mask).sum(1) / mask.sum(1)
        elif self.pooling == 'max':
            last_hidden[attention_mask == 0] = -1e9
            pooled = last_hidden.max(dim=1).values
        elif self.pooling == 'attention':
            weights = self.attention(last_hidden).squeeze(-1)
            weights[attention_mask == 0] = -1e9
            weights = torch.softmax(weights, dim=1).unsqueeze(-1)
            pooled = (last_hidden * weights).sum(1)
        elif self.pooling == 'concat_cls_mean':
            cls_token = last_hidden[:, 0]
            mask = attention_mask.unsqueeze(-1).float()
            mean_pooled = (last_hidden * mask).sum(1) / mask.sum(1)
            pooled = torch.cat([cls_token, mean_pooled], dim=1)
            # Ajuster le classifier pour 2x hidden_size
        else:
            pooled = last_hidden[:, 0]

        logits = self.classifier(pooled)
        return logits
```

### 4. Training avec Gradient Accumulation et Mixed Precision

```python
def train_one_epoch(model, loader, optimizer, scheduler, device, cfg):
    model.train()
    running_loss = 0
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss()

    optimizer.zero_grad()

    pbar = tqdm(loader, desc='Training')
    for step, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        token_type_ids = batch.get('token_type_ids')
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        with torch.cuda.amp.autocast():
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels)
            loss = loss / cfg.gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        running_loss += loss.item() * cfg.gradient_accumulation_steps
        pbar.set_postfix(loss=running_loss / (step + 1))

    return running_loss / len(loader)
```

### 5. Fine-tuning de LLM avec LoRA (PEFT)

```python
from peft import LoraConfig, get_peft_model, TaskType

def create_lora_model(model_name, num_classes):
    """Fine-tuner un LLM avec LoRA pour la classification."""

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        torch_dtype=torch.float16,
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
        bias='none',
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model
```

### 6. Text Preprocessing

```python
import re
import unicodedata

def clean_text(text):
    """Nettoyage de texte standard pour NLP."""
    if pd.isna(text):
        return ""

    text = str(text)

    # Normaliser les caractères unicode
    text = unicodedata.normalize('NFKD', text)

    # Supprimer les URLs
    text = re.sub(r'http\S+|www\.\S+', ' [URL] ', text)

    # Supprimer les mentions et hashtags (si réseau social)
    text = re.sub(r'@\w+', ' [USER] ', text)
    text = re.sub(r'#(\w+)', r' \1 ', text)

    # Normaliser les espaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def create_text_features(df, text_col):
    """Features statistiques à partir du texte."""
    df[f'{text_col}_len'] = df[text_col].str.len()
    df[f'{text_col}_word_count'] = df[text_col].str.split().str.len()
    df[f'{text_col}_unique_words'] = df[text_col].apply(lambda x: len(set(str(x).split())))
    df[f'{text_col}_avg_word_len'] = df[text_col].apply(
        lambda x: np.mean([len(w) for w in str(x).split()]) if str(x).split() else 0
    )
    df[f'{text_col}_upper_ratio'] = df[text_col].apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / (len(str(x)) + 1)
    )
    df[f'{text_col}_punct_ratio'] = df[text_col].apply(
        lambda x: sum(1 for c in str(x) if c in '!?.,;:') / (len(str(x)) + 1)
    )
    return df
```

## Approches Classiques NLP (Baselines et Ensembling)

Les approches classiques restent très utiles en compétition : baselines rapides, composants d'ensemble diversifiés, et situations GPU-limitées.

### 7. TF-IDF et CountVectorizer

```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# TF-IDF : le vectoriseur classique le plus puissant
tfidf = TfidfVectorizer(
    max_features=50000,        # Vocabulaire max
    min_df=3,                  # Ignorer les mots apparaissant dans < 3 docs
    max_df=0.95,               # Ignorer les mots dans > 95% des docs
    ngram_range=(1, 3),        # Unigrammes, bigrammes, trigrammes
    sublinear_tf=True,         # Appliquer log(1 + tf) — améliore presque toujours
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',   # Mots d'au moins 1 caractère
    dtype=np.float32,
)

X_train_tfidf = tfidf.fit_transform(train['text'])
X_test_tfidf = tfidf.transform(test['text'])
print(f"TF-IDF shape: {X_train_tfidf.shape}")

# CountVectorizer : comptage brut (utile pour Naive Bayes)
count_vec = CountVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    min_df=3,
)
X_train_count = count_vec.fit_transform(train['text'])

# TF-IDF sur caractères (capture les patterns morphologiques)
tfidf_char = TfidfVectorizer(
    max_features=50000,
    analyzer='char_wb',         # Caractères avec limites de mots
    ngram_range=(2, 6),         # Bigrammes à 6-grammes de caractères
    sublinear_tf=True,
    dtype=np.float32,
)
X_train_char = tfidf_char.fit_transform(train['text'])

# Combinaison word + char TF-IDF (souvent meilleur)
from scipy.sparse import hstack
X_train_combined = hstack([X_train_tfidf, X_train_char])
X_test_combined = hstack([
    tfidf.transform(test['text']),
    tfidf_char.transform(test['text'])
])
```

### 8. Modèles ML Classiques sur Texte

```python
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, StratifiedKFold

# --- Logistic Regression (souvent la meilleure baseline) ---
lr_model = LogisticRegression(
    C=1.0,                      # Régularisation (tuner avec Optuna)
    max_iter=1000,
    solver='lbfgs',
    n_jobs=-1,
)
scores = cross_val_score(lr_model, X_train_tfidf, y_train,
                         cv=StratifiedKFold(5, shuffle=True, random_state=42),
                         scoring='roc_auc', n_jobs=-1)
print(f"LogReg TF-IDF AUC: {scores.mean():.6f} (+/- {scores.std():.6f})")

# --- Naive Bayes (rapide, bon avec CountVectorizer) ---
# MultinomialNB : bon par défaut
nb_model = MultinomialNB(alpha=0.1)  # alpha = smoothing (tuner)

# ComplementNB : meilleur sur classes déséquilibrées
cnb_model = ComplementNB(alpha=0.5)

# --- LinearSVC (souvent compétitif avec LogReg) ---
svc_model = LinearSVC(C=1.0, max_iter=10000)

# --- LightGBM sur TF-IDF (pour ensembling) ---
# Convertir sparse → dense ou utiliser un SVD d'abord
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=300, random_state=42)
X_train_svd = svd.fit_transform(X_train_tfidf)
X_test_svd = svd.transform(X_test_tfidf)

lgb_text = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, verbose=-1)
scores_lgb = cross_val_score(lgb_text, X_train_svd, y_train,
                              cv=StratifiedKFold(5, shuffle=True, random_state=42),
                              scoring='roc_auc', n_jobs=-1)
print(f"LGB SVD-TF-IDF AUC: {scores_lgb.mean():.6f}")

# --- Ensemble classique ---
ensemble = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(C=1.0, max_iter=1000)),
        ('nb', MultinomialNB(alpha=0.1)),
        ('cnb', ComplementNB(alpha=0.5)),
    ],
    voting='soft',  # 'soft' pour moyenner les probabilités
    n_jobs=-1,
)
```

### 9. Word Embeddings (GloVe, Word2Vec)

```python
import numpy as np

# --- Charger les embeddings GloVe pré-entraînés ---
def load_glove(path, dim=300):
    """Charger les embeddings GloVe depuis un fichier texte.
    Fichiers : glove.6B.100d.txt, glove.6B.300d.txt, glove.840B.300d.txt
    """
    embeddings = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.rstrip().split(' ')
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    print(f"GloVe chargé : {len(embeddings)} mots, dim={dim}")
    return embeddings

glove = load_glove('glove.6B.300d.txt', dim=300)

# --- Charger Word2Vec (Google News 300d) ---
# pip install gensim
from gensim.models import KeyedVectors

w2v = KeyedVectors.load_word2vec_format(
    'GoogleNews-vectors-negative300.bin', binary=True
)

# --- Créer la matrice d'embeddings pour un vocabulaire donné ---
def build_embedding_matrix(word_index, embeddings, embed_dim=300):
    """Construire une matrice d'embeddings alignée avec le tokenizer."""
    vocab_size = len(word_index) + 1
    matrix = np.zeros((vocab_size, embed_dim))
    found = 0
    for word, idx in word_index.items():
        vector = embeddings.get(word)
        if vector is not None and len(vector) == embed_dim:
            matrix[idx] = vector
            found += 1
    print(f"Embeddings trouvés : {found}/{len(word_index)} ({found/len(word_index)*100:.1f}%)")
    return matrix

# --- Avec Keras Tokenizer ---
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_WORDS = 50000
MAX_LEN = 200

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(train['text'])

X_train_seq = pad_sequences(tokenizer.texts_to_sequences(train['text']), maxlen=MAX_LEN)
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(test['text']), maxlen=MAX_LEN)

embedding_matrix = build_embedding_matrix(tokenizer.word_index, glove, embed_dim=300)

# --- Features moyennes d'embeddings (baseline simple) ---
def text_to_avg_embedding(texts, embeddings, dim=300):
    """Vectoriser du texte en moyennant les embeddings des mots."""
    result = np.zeros((len(texts), dim))
    for i, text in enumerate(texts):
        words = str(text).lower().split()
        vectors = [embeddings[w] for w in words if w in embeddings]
        if vectors:
            result[i] = np.mean(vectors, axis=0)
    return result

X_train_emb = text_to_avg_embedding(train['text'], glove)
X_test_emb = text_to_avg_embedding(test['text'], glove)
# Utiliser directement avec LightGBM, XGBoost, etc.
```

### 10. RNN / LSTM / GRU / Bi-LSTM

```python
import torch
import torch.nn as nn

# --- LSTM simple ---
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_layers=2, dropout=0.3, bidirectional=True,
                 pretrained_embeddings=None):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(
                torch.tensor(pretrained_embeddings, dtype=torch.float32)
            )
            self.embedding.weight.requires_grad = False  # Freeze embeddings

        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        direction_factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * direction_factor, num_classes)

    def forward(self, x):
        # x: (batch, seq_len) d'indices de mots
        embedded = self.embedding(x)               # (batch, seq_len, embed_dim)
        output, (hidden, cell) = self.lstm(embedded)

        # Prendre le dernier hidden state (concat forward + backward si bidirectionnel)
        if self.lstm.bidirectional:
            hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden_cat = hidden[-1]

        out = self.dropout(hidden_cat)
        logits = self.fc(out)
        return logits

# --- GRU (plus rapide que LSTM, performances similaires) ---
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_layers=2, dropout=0.3, bidirectional=True,
                 pretrained_embeddings=None):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(
                torch.tensor(pretrained_embeddings, dtype=torch.float32)
            )
            self.embedding.weight.requires_grad = False

        self.gru = nn.GRU(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        direction_factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * direction_factor, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded)

        if self.gru.bidirectional:
            hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden_cat = hidden[-1]

        out = self.dropout(hidden_cat)
        return self.fc(out)

# --- Instanciation avec GloVe ---
model = LSTMClassifier(
    vocab_size=len(tokenizer.word_index) + 1,
    embed_dim=300,
    hidden_dim=128,
    num_classes=2,
    num_layers=2,
    dropout=0.3,
    bidirectional=True,
    pretrained_embeddings=embedding_matrix,
)

# --- Avec Keras/TensorFlow ---
import tensorflow as tf
from tensorflow.keras.layers import (
    Embedding, LSTM, GRU, Bidirectional, Dense, Dropout,
    GlobalMaxPooling1D, GlobalAveragePooling1D, Concatenate
)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

def build_bilstm_keras(vocab_size, embed_dim, max_len, num_classes,
                       embedding_matrix=None):
    model = Sequential([
        Embedding(vocab_size, embed_dim, weights=[embedding_matrix] if embedding_matrix is not None else None,
                  input_length=max_len, trainable=embedding_matrix is None),
        Bidirectional(LSTM(128, return_sequences=True, dropout=0.2)),
        Bidirectional(LSTM(64, dropout=0.2)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid'),
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
        metrics=['accuracy']
    )
    return model
```

### 11. Preprocessing Avancé NLP

```python
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer, SnowballStemmer

# Téléchargements NLTK (une seule fois)
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')

# --- Stopwords ---
STOP_WORDS = set(stopwords.words('english'))
# Ajouter des mots custom si nécessaire
# STOP_WORDS.update(['also', 'would', 'could'])

# --- Stemming (rapide, agressif) ---
stemmer = PorterStemmer()
# Ou SnowballStemmer('french') pour le français

def stem_text(text):
    return ' '.join([stemmer.stem(w) for w in text.split()])

# --- Lemmatization (plus précis, plus lent) ---
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(w) for w in text.split()])

# --- Pipeline de preprocessing complet ---
def preprocess_for_classical_ml(text, remove_stopwords=True, use_stemming=False,
                                 use_lemmatization=True):
    """Preprocessing pour TF-IDF et modèles classiques.

    IMPORTANT : NE PAS utiliser ce preprocessing pour les Transformers !
    Les Transformers ont leur propre tokenizer et bénéficient du texte brut.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)         # Supprimer URLs
    text = re.sub(r'<.*?>', '', text)                     # Supprimer HTML
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)             # Garder lettres seulement
    text = re.sub(r'\s+', ' ', text).strip()              # Normaliser espaces

    words = text.split()

    if remove_stopwords:
        words = [w for w in words if w not in STOP_WORDS]

    if use_stemming:
        words = [stemmer.stem(w) for w in words]
    elif use_lemmatization:
        words = [lemmatizer.lemmatize(w) for w in words]

    return ' '.join(words)

# --- Quand NE PAS préprocesser ---
# 1. Transformers (BERT, DeBERTa, etc.) : NE PAS supprimer stopwords,
#    NE PAS stemmer/lemmatiser. Le tokenizer gère tout.
# 2. Modèles avec embeddings pré-entraînés : preprocessing minimal,
#    garder la casse et la ponctuation peut être utile.
# 3. Pour TF-IDF + ML classiques : preprocessing complet recommandé.
```

### 12. EDA Spécifique NLP

```python
from collections import Counter

def nlp_eda(df, text_col, target_col=None, top_n=20):
    """Analyse exploratoire spécifique au texte."""

    print("=" * 50)
    print(f"EDA NLP : colonne '{text_col}'")
    print("=" * 50)

    # Statistiques de base
    df['_len'] = df[text_col].str.len()
    df['_words'] = df[text_col].str.split().str.len()
    df['_unique'] = df[text_col].apply(lambda x: len(set(str(x).lower().split())))

    stats = df[['_len', '_words', '_unique']].describe()
    print("\nStatistiques textuelles :")
    print(stats.round(1))

    # Distribution des longueurs
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    for i, (col, title) in enumerate([
        ('_len', 'Longueur (caractères)'),
        ('_words', 'Nombre de mots'),
        ('_unique', 'Mots uniques')
    ]):
        if target_col and df[target_col].nunique() <= 10:
            sns.histplot(data=df, x=col, hue=target_col, bins=50, ax=axes[i],
                         kde=True, alpha=0.5)
        else:
            sns.histplot(data=df, x=col, bins=50, ax=axes[i], kde=True)
        axes[i].set_title(title)
    plt.tight_layout()
    plt.show()

    # Mots les plus fréquents
    all_words = ' '.join(df[text_col].astype(str)).lower().split()
    word_freq = Counter(all_words).most_common(top_n)
    print(f"\nTop {top_n} mots les plus fréquents :")
    for word, count in word_freq:
        print(f"  {word}: {count}")

    # Mots fréquents par classe
    if target_col and df[target_col].nunique() <= 10:
        fig, axes = plt.subplots(1, df[target_col].nunique(), figsize=(8 * df[target_col].nunique(), 5))
        if df[target_col].nunique() == 1:
            axes = [axes]
        for i, cls in enumerate(sorted(df[target_col].unique())):
            subset = df[df[target_col] == cls]
            words = ' '.join(subset[text_col].astype(str)).lower().split()
            freq = Counter(words).most_common(top_n)
            axes[i].barh([w for w, _ in freq], [c for _, c in freq], color='steelblue')
            axes[i].set_title(f'Classe {cls} (n={len(subset)})')
            axes[i].invert_yaxis()
        plt.suptitle('Mots fréquents par classe', fontsize=14)
        plt.tight_layout()
        plt.show()

    # WordCloud (optionnel)
    try:
        from wordcloud import WordCloud
        fig, ax = plt.subplots(figsize=(12, 6))
        wc = WordCloud(width=800, height=400, background_color='white',
                       max_words=200, colormap='viridis')
        wc.generate(' '.join(df[text_col].astype(str)))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('WordCloud')
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("(WordCloud non installé — pip install wordcloud)")

    # Nettoyage des colonnes temporaires
    df.drop(columns=['_len', '_words', '_unique'], inplace=True)

    # Analyse de la distribution max_length pour les Transformers
    lengths = df[text_col].str.split().str.len()
    for pct in [90, 95, 99, 100]:
        val = lengths.quantile(pct / 100)
        print(f"  P{pct} longueur : {val:.0f} mots")
    print(f"\n→ max_length recommandé pour tokenizer : ~{int(lengths.quantile(0.95) * 1.3)} tokens")
```

## Quand Utiliser Chaque Approche

| Approche | Quand l'utiliser | Avantage | Inconvénient |
|---|---|---|---|
| TF-IDF + LogReg | Baseline rapide, GPU limité | Rapide, interprétable | Plafonne vite |
| TF-IDF + Naive Bayes | Texte court, classes multiples | Très rapide | Moins précis |
| GloVe + Bi-LSTM | Budget GPU moyen | Bon compromis coût/perf | Entraînement plus long |
| Word2Vec + GRU | Séquences longues | Rapide vs LSTM | Perd parfois du contexte |
| DeBERTa fine-tuning | Meilleur score possible | SOTA | GPU nécessaire |
| LLM + LoRA | Très gros contexte, peu de données | Puissant avec peu de données | GPU important |
| TF-IDF + SVD + GBDT | Composant d'ensemble diversifié | Décorrélé des Transformers | Seul, moins fort |

### Stratégie d'Ensemble NLP Gold Medal

```python
# L'ensemble idéal en NLP combine des modèles à approches différentes :
#
# 1. DeBERTa-v3-large (mean pooling) — fold 0-4
# 2. DeBERTa-v3-large (attention pooling) — fold 0-4
# 3. RoBERTa-large — fold 0-4
# 4. Llama-3.1-8B + LoRA — fold 0-4
# 5. TF-IDF (word+char) + LogReg — CV prédictions
# 6. GloVe + Bi-LSTM — fold 0-4
#
# Ensemble final : weighted average ou stacking via Ridge/LightGBM
# Les modèles classiques (5, 6) apportent de la diversité même si
# leur score individuel est plus faible.
```

## Stratégies Gold Medal NLP

1. **DeBERTa-v3-large comme baseline** : presque toujours le meilleur pour la classification
2. **Multi-modèles** : DeBERTa + RoBERTa + ELECTRA + LLM (LoRA)
3. **Pooling diversifié** : CLS, mean, attention, concat
4. **Max length adapté** : analyser la distribution des longueurs
5. **Multi-fold** : 5 folds minimum, chaque fold = 1 modèle pour l'ensemble
6. **Pseudo-labeling** : après une première passe, ajouter des pseudo-labels
7. **Reinitialize top layers** : réinitialiser les dernières couches du transformer
8. **Layer-wise LR decay** : LR plus faible pour les couches profondes
9. **AWP (Adversarial Weight Perturbation)** : perturbation adversariale des poids
10. **Stacking** : OOF predictions des transformers comme features pour GBDT

```python
# Layer-wise LR decay
def get_optimizer_params(model, lr, weight_decay, llrd=0.9):
    """Learning rate decay par couche."""
    opt_params = []
    no_decay = ['bias', 'LayerNorm.weight']

    for i, (name, param) in enumerate(model.backbone.named_parameters()):
        layer_num = None
        for part in name.split('.'):
            if part.isdigit():
                layer_num = int(part)
                break

        layer_lr = lr * (llrd ** (model.backbone.config.num_hidden_layers - layer_num)) \
                   if layer_num is not None else lr * (llrd ** 0)

        opt_params.append({
            'params': [param],
            'lr': layer_lr,
            'weight_decay': 0.0 if any(nd in name for nd in no_decay) else weight_decay
        })

    # Classifier avec LR standard
    opt_params.append({
        'params': model.classifier.parameters(),
        'lr': lr,
        'weight_decay': weight_decay
    })

    return opt_params
```

Adapte TOUJOURS au type de tâche NLP et aux données spécifiques de la compétition.
