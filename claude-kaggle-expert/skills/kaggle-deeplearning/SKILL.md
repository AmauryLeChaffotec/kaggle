---
name: kaggle-deeplearning
description: Deep Learning pour données tabulaires en compétitions Kaggle. Utiliser quand l'utilisateur veut utiliser TabNet, FT-Transformer, SAINT, entity embeddings, autoencoders, ou combiner NN et GBDT.
argument-hint: <type de modèle DL ou description du problème>
---

# Deep Learning for Tabular Data Expert - Kaggle Gold Medal

Tu es un expert en deep learning pour données tabulaires en compétitions Kaggle. Les top solutions 2024-2025 combinent systématiquement GBDT + Neural Networks pour la diversité d'ensemble.

## Philosophie

- **DL tabulaire ≠ remplacer GBDT** : c'est un COMPLÉMENT pour la diversité d'ensemble
- **GBDT reste roi sur tabulaire** pour un modèle seul. DL brille dans l'ensemble
- **Entity embeddings** sont la contribution #1 du DL au tabulaire
- **Préprocessing différent** : NN ont besoin de normalisation, contrairement aux GBDT

## Quand Utiliser le DL sur Tabulaire ?

```
Utiliser DL tabulaire quand :
✓ Tu as besoin de diversité d'ensemble (corrélation GBDT > 0.97)
✓ Le dataset est assez grand (>10K lignes)
✓ Tu as beaucoup de features catégorielles à haute cardinalité
✓ Les interactions entre features sont complexes/non-linéaires
✓ Tu veux extraire des embeddings comme features pour GBDT

NE PAS utiliser quand :
✗ Dataset petit (<5K lignes) — risque d'overfitting
✗ Tu n'as que des features numériques simples
✗ Pas de GPU disponible
✗ Le temps est limité (GBDT est 10x plus rapide à itérer)
```

## 1. TabNet (Google Research)

```python
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from sklearn.preprocessing import LabelEncoder
import torch

def train_tabnet(X_train, y_train, X_val, y_val, cat_cols, task='classification'):
    """TabNet : attention-based, feature selection intégré, interprétable."""

    # Encoder les catégorielles
    cat_idxs = [X_train.columns.get_loc(c) for c in cat_cols]
    cat_dims = [X_train[c].nunique() + 1 for c in cat_cols]  # +1 pour unknown

    if task == 'classification':
        model = TabNetClassifier(
            n_d=32, n_a=32,          # Dimension des layers
            n_steps=5,                # Nombre d'étapes d'attention
            gamma=1.5,                # Coefficient de sparsité
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=4,            # Dimension des embeddings catégoriels
            optimizer_fn=torch.optim.Adam,
            optimizer_params={'lr': 2e-2},
            scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
            scheduler_params={'T_0': 50, 'eta_min': 1e-4},
            mask_type='entmax',       # 'sparsemax' ou 'entmax'
            verbose=10,
            seed=42,
        )
    else:
        model = TabNetRegressor(
            n_d=32, n_a=32, n_steps=5, gamma=1.5,
            cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=4,
            optimizer_fn=torch.optim.Adam,
            optimizer_params={'lr': 2e-2},
            verbose=10, seed=42,
        )

    model.fit(
        X_train.values, y_train.values,
        eval_set=[(X_val.values, y_val.values)],
        eval_metric=['auc'] if task == 'classification' else ['rmse'],
        max_epochs=200,
        patience=30,
        batch_size=1024,
        virtual_batch_size=256,
        drop_last=False,
    )

    # Feature importance (intégrée dans TabNet)
    feat_imp = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return model, feat_imp
```

## 2. FT-Transformer (Feature Tokenizer + Transformer)

```python
import torch
import torch.nn as nn

class FTTransformer(nn.Module):
    """FT-Transformer : tokenize chaque feature puis applique un Transformer.
    State-of-the-art sur de nombreux benchmarks tabulaires.
    """

    def __init__(self, n_num_features, cat_cardinalities, d_model=192,
                 n_heads=8, n_layers=3, d_ffn=256, dropout=0.2,
                 n_classes=2):
        super().__init__()

        self.d_model = d_model

        # Numerical feature tokenizer
        self.num_tokenizer = nn.Linear(1, d_model)
        self.num_bias = nn.Parameter(torch.zeros(n_num_features, d_model))

        # Categorical embeddings
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card + 1, d_model) for card in cat_cardinalities
        ])

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ffn,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_classes)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_num, x_cat):
        batch_size = x_num.size(0)

        # Tokenize numerical features
        num_tokens = self.num_tokenizer(x_num.unsqueeze(-1)) + self.num_bias

        # Tokenize categorical features
        cat_tokens = []
        for i, emb in enumerate(self.cat_embeddings):
            cat_tokens.append(emb(x_cat[:, i]))
        cat_tokens = torch.stack(cat_tokens, dim=1) if cat_tokens else torch.empty(batch_size, 0, self.d_model)

        # Concatenate all tokens + CLS
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, num_tokens, cat_tokens], dim=1)
        tokens = self.dropout(tokens)

        # Transformer
        output = self.transformer(tokens)

        # Classification from CLS token
        cls_output = output[:, 0]
        return self.head(cls_output)
```

## 3. SAINT (Self-Attention and Intersample Attention)

```python
class SAINTLayer(nn.Module):
    """SAINT : attention entre features ET entre échantillons."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        # Self-attention (inter-feature)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        # Intersample attention
        self.intersample_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (batch, n_features, d_model)

        # Self-attention (between features for each sample)
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)

        # Intersample attention (between samples for each feature)
        # Reshape: (n_features, batch, d_model)
        x_t = x.transpose(0, 1)
        inter_out, _ = self.intersample_attn(x_t, x_t, x_t)
        x_t = self.norm2(x_t + inter_out)
        x = x_t.transpose(0, 1)

        # FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)

        return x
```

## 4. Entity Embeddings (le plus impactant)

```python
class EntityEmbeddingModel(nn.Module):
    """Entity embeddings : transformer les catégorielles en vecteurs denses.
    Les embeddings entraînés peuvent être réutilisés comme features pour GBDT !
    """

    def __init__(self, cat_cardinalities, emb_dims, n_num_features, hidden_dims=[256, 128], n_classes=2, dropout=0.3):
        super().__init__()

        # Embeddings catégoriels
        self.embeddings = nn.ModuleList([
            nn.Embedding(card + 1, dim) for card, dim in zip(cat_cardinalities, emb_dims)
        ])
        self.emb_dropout = nn.Dropout(dropout)

        # Input dim = sum(emb_dims) + n_num_features
        total_emb_dim = sum(emb_dims)
        input_dim = total_emb_dim + n_num_features

        # MLP
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, n_classes))
        self.mlp = nn.Sequential(*layers)

        # Batch norm pour les numériques
        self.num_bn = nn.BatchNorm1d(n_num_features)

    def forward(self, x_num, x_cat):
        # Embed categoricals
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        embs = torch.cat(embs, dim=1)
        embs = self.emb_dropout(embs)

        # Normalize numericals
        x_num = self.num_bn(x_num)

        # Concat and predict
        x = torch.cat([embs, x_num], dim=1)
        return self.mlp(x)

    def get_embeddings(self, x_cat):
        """Extraire les embeddings pour les réutiliser dans GBDT."""
        with torch.no_grad():
            embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            return torch.cat(embs, dim=1)

# Calcul de la dimension d'embedding optimale
def embedding_dim(cardinality, max_dim=50):
    """Règle empirique : min(50, (cardinality+1)//2) ou min(600, round(1.6 * card^0.56))"""
    return min(max_dim, (cardinality + 1) // 2)
```

## 5. Autoencoder pour Feature Extraction

```python
class TabularAutoencoder(nn.Module):
    """Autoencoder débruiteur pour extraire des features latentes.
    Les features du bottleneck peuvent être ajoutées au GBDT.
    """

    def __init__(self, input_dim, encoding_dim=32, hidden_dims=[128, 64]):
        super().__init__()

        # Encoder
        encoder_layers = []
        prev = input_dim
        for dim in hidden_dims:
            encoder_layers.extend([nn.Linear(prev, dim), nn.ReLU(), nn.Dropout(0.3)])
            prev = dim
        encoder_layers.append(nn.Linear(prev, encoding_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        prev = encoding_dim
        for dim in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(prev, dim), nn.ReLU(), nn.Dropout(0.3)])
            prev = dim
        decoder_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        """Extraire les features latentes."""
        with torch.no_grad():
            return self.encoder(x)

def train_autoencoder_features(train_df, test_df, num_features, encoding_dim=32, epochs=100):
    """Entraîner un autoencoder et extraire des features pour GBDT."""
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader, TensorDataset

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[num_features].fillna(0))
    X_test = scaler.transform(test_df[num_features].fillna(0))

    # Train autoencoder
    model = TabularAutoencoder(len(num_features), encoding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dataset = TensorDataset(torch.FloatTensor(X_train))
    loader = DataLoader(dataset, batch_size=512, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for (batch,) in loader:
            # Denoising : ajouter du bruit
            noisy = batch + 0.1 * torch.randn_like(batch)
            recon = model(noisy)
            loss = nn.MSELoss()(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # Extraire les features
    model.eval()
    train_encoded = model.encode(torch.FloatTensor(X_train)).numpy()
    test_encoded = model.encode(torch.FloatTensor(X_test)).numpy()

    # Ajouter au DataFrame
    for i in range(encoding_dim):
        train_df[f'ae_feat_{i}'] = train_encoded[:, i]
        test_df[f'ae_feat_{i}'] = test_encoded[:, i]

    print(f"Added {encoding_dim} autoencoder features")
    return train_df, test_df, model
```

## 6. Training Recipe Standard

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class TabularDataset(Dataset):
    def __init__(self, X_num, X_cat, y=None):
        self.X_num = torch.FloatTensor(X_num.values if hasattr(X_num, 'values') else X_num)
        self.X_cat = torch.LongTensor(X_cat.values if hasattr(X_cat, 'values') else X_cat)
        self.y = torch.FloatTensor(y.values if hasattr(y, 'values') else y) if y is not None else None

    def __len__(self):
        return len(self.X_num)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_num[idx], self.X_cat[idx], self.y[idx]
        return self.X_num[idx], self.X_cat[idx]

def train_nn_fold(model, train_loader, val_loader, epochs=100, lr=1e-3,
                  patience=10, device='cuda'):
    """Training loop standard avec early stopping et mixed precision."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for x_num, x_cat, y in train_loader:
            x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)

            with torch.cuda.amp.autocast():
                pred = model(x_num, x_cat).squeeze()
                loss = criterion(pred, y)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0
        val_preds = []
        with torch.no_grad():
            for x_num, x_cat, y in val_loader:
                x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
                pred = model(x_num, x_cat).squeeze()
                val_loss += criterion(pred, y).item()
                val_preds.extend(torch.sigmoid(pred).cpu().numpy())

        val_loss /= len(val_loader)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model
```

## 7. Stratégie GBDT + NN Ensemble

```python
def gbdt_nn_pipeline(train, test, target, num_cols, cat_cols, n_folds=5):
    """Pipeline complet GBDT + NN pour diversité maximale d'ensemble.

    Stratégie :
    1. Train GBDT (LightGBM, XGBoost, CatBoost) → OOF + test preds
    2. Train NN (TabNet ou FT-Transformer) → OOF + test preds
    3. Optionnel : extraire embeddings du NN → features pour un 2ème GBDT
    4. Ensemble : rank average des OOF
    """
    # Le NN apporte typiquement 0.001-0.005 de gain d'ensemble
    # grâce à sa corrélation ~0.95 avec GBDT (vs ~0.99 entre GBDTs)

    print("=== Step 1: GBDT Models ===")
    # ... LightGBM, XGBoost, CatBoost OOF training ...

    print("\n=== Step 2: Neural Network ===")
    # ... TabNet ou FT-Transformer OOF training ...

    print("\n=== Step 3: Ensemble ===")
    # Rank average : robuste aux différences d'échelle proba vs logit
    # Poids typiques : GBDT ~0.7, NN ~0.3
    pass

# Corrélation typique entre modèles tabulaires :
# LGB vs XGB    : 0.98-0.99  (très corrélé)
# LGB vs CatB   : 0.97-0.99  (très corrélé)
# GBDT vs TabNet : 0.93-0.96  (bonne diversité !)
# GBDT vs MLP    : 0.92-0.95  (bonne diversité !)
```

## Règles d'Or du DL Tabulaire

1. **GBDT first** : toujours avoir une baseline GBDT avant de tenter du DL
2. **Entity embeddings** : la technique DL la plus utile, même pour enrichir GBDT
3. **Normalisation obligatoire** : StandardScaler ou RobustScaler sur les numériques
4. **Batch size large** (512-2048) pour le tabulaire
5. **AdamW > Adam** : weight decay aide contre l'overfitting
6. **Mixed precision** pour accélérer x2 sur GPU
7. **Autoencoder features** : DAE sur train+test (pas de leakage car unsupervised)
8. **Le gain est dans l'ensemble** : NN seul < GBDT seul, mais GBDT+NN > GBDT seul

## Rapport de Sortie (OBLIGATOIRE)

À la fin de l'entraînement DL, TOUJOURS sauvegarder :
1. Rapport dans : `reports/deeplearning/YYYY-MM-DD_<model>.md` (architecture, params, scores, training curves)
2. OOF predictions dans : `artifacts/oof_<model>_v<N>.parquet`
3. Test predictions dans : `artifacts/test_<model>_v<N>.parquet`
4. Ajouter une ligne dans `runs.csv`
5. Confirmer à l'utilisateur les chemins sauvegardés
