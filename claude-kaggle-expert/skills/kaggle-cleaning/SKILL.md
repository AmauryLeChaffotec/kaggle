---
name: kaggle-cleaning
description: Nettoyage et preprocessing des données pour compétitions Kaggle. Utiliser quand l'utilisateur veut nettoyer ses données, traiter les valeurs manquantes, les outliers, les doublons, corriger les types, normaliser ou transformer ses features.
argument-hint: <chemin_du_dataset ou description du problème>
---

# Data Cleaning & Preprocessing Expert - Kaggle Gold Medal

Tu es un expert en nettoyage de données pour compétitions Kaggle. Le nettoyage représente 30-40% du travail réel d'un data scientist et fait souvent la différence entre une bonne et une excellente solution.

## Philosophie

- **Ne JAMAIS supprimer de données sans justification** : chaque ligne supprimée = information perdue
- **Le nettoyage DOIT être reproductible** : tout dans des fonctions, jamais de modifications manuelles
- **Train et test doivent subir le MÊME nettoyage** (sauf target encoding)
- **Documenter chaque décision** : pourquoi ce choix plutôt qu'un autre

## Workflow de Nettoyage Complet (10 étapes)

### Étape 1 : Audit Initial des Données

```python
def data_audit(df, name='Dataset'):
    """Audit complet d'un DataFrame."""
    print(f"{'='*60}")
    print(f"AUDIT : {name}")
    print(f"{'='*60}")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print(f"\nDuplicates: {df.duplicated().sum()} ({df.duplicated().sum()/len(df)*100:.2f}%)")

    # Types
    print(f"\nColumn types:")
    for dtype, count in df.dtypes.value_counts().items():
        print(f"  {dtype}: {count}")

    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'missing': missing[missing > 0],
        'pct': missing_pct[missing > 0]
    }).sort_values('pct', ascending=False)

    if len(missing_df) > 0:
        print(f"\nMissing values ({len(missing_df)} columns):")
        for col, row in missing_df.iterrows():
            print(f"  {col}: {row['missing']:,} ({row['pct']}%)")
    else:
        print("\nNo missing values!")

    # Unique values
    print(f"\nUnique value counts:")
    for col in df.columns:
        n_unique = df[col].nunique()
        if n_unique <= 10:
            print(f"  {col}: {n_unique} unique → {df[col].value_counts().index.tolist()}")
        else:
            print(f"  {col}: {n_unique} unique")

    return missing_df

audit_train = data_audit(train, 'Train')
audit_test = data_audit(test, 'Test')
```

### Étape 2 : Gestion des Valeurs Manquantes

```python
# === STRATÉGIE DE MISSING VALUES ===
# Décision tree :
# 1. >70% missing → DROP la colonne (sauf si le pattern de missing est informatif)
# 2. MCAR (Missing Completely At Random) → Imputation simple
# 3. MAR (Missing At Random) → Imputation conditionnelle
# 4. MNAR (Missing Not At Random) → Le missing EST l'information → créer indicateur

# --- Détecter le type de missing ---
def analyze_missing_pattern(df, target=None):
    """Analyse si le missing est informatif."""
    missing_cols = df.columns[df.isnull().any()].tolist()
    results = {}

    for col in missing_cols:
        info = {
            'pct_missing': df[col].isnull().mean() * 100,
            'missing_indicator_useful': False
        }

        # Test si le pattern de missing corrèle avec la target
        if target is not None and target in df.columns:
            df_temp = df.copy()
            df_temp[f'{col}_is_null'] = df_temp[col].isnull().astype(int)
            if df_temp[target].dtype in ['int64', 'float64']:
                corr = df_temp[f'{col}_is_null'].corr(df_temp[target])
                info['target_corr'] = abs(corr)
                info['missing_indicator_useful'] = abs(corr) > 0.02

        results[col] = info

    return pd.DataFrame(results).T

# --- Imputation avancée ---
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def smart_impute(train_df, test_df, num_cols, cat_cols, strategy='auto'):
    """Imputation intelligente avec fit sur train, transform sur test."""
    df_all = pd.concat([train_df, test_df], axis=0)

    # Numériques
    for col in num_cols:
        pct_missing = train_df[col].isnull().mean()
        if pct_missing == 0:
            continue

        # Créer indicateur de missing si informatif
        train_df[f'{col}_is_null'] = train_df[col].isnull().astype(int)
        test_df[f'{col}_is_null'] = test_df[col].isnull().astype(int)

        if strategy == 'auto':
            if pct_missing < 0.05:
                fill_val = train_df[col].median()
            elif pct_missing < 0.30:
                # KNN imputation
                imputer = KNNImputer(n_neighbors=5)
                train_df[col] = imputer.fit_transform(train_df[[col]])
                test_df[col] = imputer.transform(test_df[[col]])
                continue
            else:
                fill_val = train_df[col].median()
        elif strategy == 'median':
            fill_val = train_df[col].median()
        elif strategy == 'mean':
            fill_val = train_df[col].mean()

        train_df[col] = train_df[col].fillna(fill_val)
        test_df[col] = test_df[col].fillna(fill_val)

    # Catégorielles
    for col in cat_cols:
        if train_df[col].isnull().any():
            train_df[f'{col}_is_null'] = train_df[col].isnull().astype(int)
            test_df[f'{col}_is_null'] = test_df[col].isnull().astype(int)
            mode_val = train_df[col].mode()[0]
            train_df[col] = train_df[col].fillna(mode_val)
            test_df[col] = test_df[col].fillna(mode_val)

    return train_df, test_df

# --- Iterative Imputer (le plus puissant) ---
def iterative_impute(train_df, test_df, num_cols):
    """Imputation itérative (MICE) - le plus précis pour MAR."""
    imputer = IterativeImputer(
        max_iter=10,
        random_state=42,
        initial_strategy='median'
    )
    train_df[num_cols] = imputer.fit_transform(train_df[num_cols])
    test_df[num_cols] = imputer.transform(test_df[num_cols])
    return train_df, test_df
```

### Étape 3 : Détection et Traitement des Outliers

```python
import scipy.stats as stats

def detect_outliers(df, num_cols, method='iqr', threshold=1.5):
    """Détecte les outliers avec plusieurs méthodes."""
    outlier_report = {}

    for col in num_cols:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        elif method == 'zscore':
            z = np.abs(stats.zscore(df[col].dropna()))
            outliers = (z > 3).sum()
        elif method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            iso = IsolationForest(contamination=0.05, random_state=42)
            preds = iso.fit_predict(df[[col]].dropna())
            outliers = (preds == -1).sum()

        outlier_report[col] = {
            'n_outliers': outliers,
            'pct': outliers / len(df) * 100,
            'min': df[col].min(),
            'max': df[col].max(),
            'mean': df[col].mean(),
            'median': df[col].median()
        }

    return pd.DataFrame(outlier_report).T.sort_values('pct', ascending=False)

# --- Traitement des outliers ---
# ATTENTION : En compétition Kaggle, ne JAMAIS supprimer des outliers du test !
# Les outliers dans le train peuvent être informatifs

def clip_outliers(df, col, lower_pct=0.01, upper_pct=0.99):
    """Winsorization : clipper aux percentiles."""
    lower = df[col].quantile(lower_pct)
    upper = df[col].quantile(upper_pct)
    df[col] = df[col].clip(lower, upper)
    return df

def log_transform_skewed(df, num_cols, skew_threshold=1.0):
    """Log1p transform pour les features très skewed."""
    skewed = []
    for col in num_cols:
        skewness = df[col].skew()
        if abs(skewness) > skew_threshold and df[col].min() >= 0:
            df[col] = np.log1p(df[col])
            skewed.append((col, skewness))
    print(f"Log-transformed {len(skewed)} columns: {[s[0] for s in skewed]}")
    return df
```

### Étape 4 : Gestion des Doublons

```python
def handle_duplicates(df, subset=None, keep='first'):
    """Détecte et traite les doublons."""
    # Doublons exacts
    n_exact = df.duplicated(subset=subset).sum()
    print(f"Exact duplicates: {n_exact} ({n_exact/len(df)*100:.2f}%)")

    if n_exact > 0:
        # Vérifier si les doublons ont des targets différentes
        if 'target' in df.columns and subset is not None:
            dupes = df[df.duplicated(subset=subset, keep=False)]
            conflicting = dupes.groupby(subset)['target'].nunique()
            n_conflicting = (conflicting > 1).sum()
            print(f"  Conflicting targets in duplicates: {n_conflicting}")

        df = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
        print(f"  After dedup: {df.shape[0]} rows")

    return df

# Near-duplicates (pour le texte)
def find_near_duplicates_text(df, text_col, threshold=0.9):
    """Trouve les near-duplicates dans une colonne texte."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df[text_col].fillna(''))
    cos_sim = cosine_similarity(tfidf_matrix)
    np.fill_diagonal(cos_sim, 0)

    pairs = []
    for i in range(len(cos_sim)):
        for j in range(i+1, len(cos_sim)):
            if cos_sim[i][j] > threshold:
                pairs.append((i, j, cos_sim[i][j]))

    print(f"Found {len(pairs)} near-duplicate pairs (threshold={threshold})")
    return pairs
```

### Étape 5 : Correction des Types de Données

```python
def fix_data_types(df):
    """Corrige automatiquement les types de données."""
    for col in df.columns:
        # String qui devrait être numérique
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col].str.replace(',', '.').str.strip())
                print(f"  {col}: object → numeric")
                continue
            except (ValueError, AttributeError):
                pass

            # Dates
            try:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
                print(f"  {col}: object → datetime")
                continue
            except (ValueError, TypeError):
                pass

            # Boolean
            unique_vals = set(df[col].dropna().str.lower().unique())
            if unique_vals.issubset({'true', 'false', 'yes', 'no', '0', '1', 'y', 'n'}):
                mapping = {'true': 1, 'false': 0, 'yes': 1, 'no': 0,
                           '1': 1, '0': 0, 'y': 1, 'n': 0}
                df[col] = df[col].str.lower().map(mapping)
                print(f"  {col}: object → boolean")

    return df
```

### Étape 6 : Nettoyage des Données Textuelles

```python
import re

def clean_text_column(df, col):
    """Nettoyage standard d'une colonne texte."""
    df[col] = df[col].astype(str)
    df[col] = df[col].str.strip()
    df[col] = df[col].str.lower()
    # Supprimer caractères spéciaux excessifs
    df[col] = df[col].apply(lambda x: re.sub(r'[^\w\s]', ' ', x))
    # Supprimer espaces multiples
    df[col] = df[col].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    return df

def standardize_categories(df, col):
    """Standardise les catégories (fuzzy matching)."""
    from fuzzywuzzy import fuzz, process

    unique_vals = df[col].dropna().unique()
    mapping = {}

    for val in unique_vals:
        if val not in mapping:
            matches = process.extract(val, unique_vals, scorer=fuzz.ratio, limit=5)
            similar = [m[0] for m in matches if m[1] > 85 and m[0] != val]
            for s in similar:
                if s not in mapping:
                    mapping[s] = val  # Mapper vers la première occurrence

    if mapping:
        print(f"  {col}: {len(mapping)} values standardized")
        df[col] = df[col].replace(mapping)
    return df
```

### Étape 7 : Gestion de l'Encodage des Caractères

```python
def fix_encoding(filepath):
    """Détecte et corrige l'encodage d'un fichier."""
    import chardet

    with open(filepath, 'rb') as f:
        raw = f.read(100000)
        result = chardet.detect(raw)
        print(f"Detected encoding: {result['encoding']} (confidence: {result['confidence']:.2f})")

    return pd.read_csv(filepath, encoding=result['encoding'])

# Fixer les caractères mal encodés dans un DataFrame
def fix_bad_chars(df, col):
    """Corrige les caractères mal encodés."""
    replacements = {
        'â€™': "'", 'â€œ': '"', 'â€': '"',
        'Ã©': 'é', 'Ã¨': 'è', 'Ã ': 'à',
        'Ã§': 'ç', 'Ã¢': 'â', 'Ã®': 'î',
    }
    for old, new in replacements.items():
        df[col] = df[col].str.replace(old, new, regex=False)
    return df
```

### Étape 8 : Scaling et Normalisation

```python
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler,
    QuantileTransformer, PowerTransformer
)

# === Guide de choix du scaler ===
# StandardScaler    → Distribution ~normale, pas d'outliers majeurs
# RobustScaler      → Données avec outliers (utilise médiane/IQR)
# MinMaxScaler      → Borner entre 0-1, pour NN/KNN/SVM
# QuantileTransformer → Force une distribution uniforme ou normale
# PowerTransformer  → Box-Cox (>0) ou Yeo-Johnson (tout), réduit le skew

def smart_scale(train_df, test_df, num_cols, method='auto'):
    """Scaling intelligent avec fit sur train, transform sur test."""
    scalers = {}

    for col in num_cols:
        skewness = abs(train_df[col].skew())
        has_outliers = (detect_outliers(train_df, [col]).iloc[0]['pct'] > 5)

        if method == 'auto':
            if has_outliers:
                scaler = RobustScaler()
            elif skewness > 2:
                scaler = PowerTransformer(method='yeo-johnson')
            else:
                scaler = StandardScaler()
        else:
            scaler_map = {
                'standard': StandardScaler(),
                'robust': RobustScaler(),
                'minmax': MinMaxScaler(),
                'quantile': QuantileTransformer(output_distribution='normal', random_state=42),
                'power': PowerTransformer(method='yeo-johnson')
            }
            scaler = scaler_map[method]

        train_df[col] = scaler.fit_transform(train_df[[col]])
        test_df[col] = scaler.transform(test_df[[col]])
        scalers[col] = scaler

    return train_df, test_df, scalers
```

### Étape 9 : Validation Train/Test Consistency

```python
def validate_train_test_consistency(train, test, features):
    """Vérifie la cohérence entre train et test."""
    issues = []

    # Colonnes manquantes
    missing_in_test = set(features) - set(test.columns)
    if missing_in_test:
        issues.append(f"Columns in train but not test: {missing_in_test}")

    # Types différents
    for col in features:
        if col in test.columns:
            if train[col].dtype != test[col].dtype:
                issues.append(f"{col}: train={train[col].dtype}, test={test[col].dtype}")

    # Distributions très différentes (drift)
    for col in features:
        if col in test.columns and train[col].dtype in ['float64', 'int64']:
            train_mean = train[col].mean()
            test_mean = test[col].mean()
            if train_mean != 0:
                drift = abs(test_mean - train_mean) / abs(train_mean)
                if drift > 0.5:
                    issues.append(f"{col}: drift={drift:.2f} (train_mean={train_mean:.3f}, test_mean={test_mean:.3f})")

    # Catégories unseen dans test
    cat_cols = train.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if col in test.columns:
            unseen = set(test[col].dropna().unique()) - set(train[col].dropna().unique())
            if unseen:
                issues.append(f"{col}: {len(unseen)} unseen categories in test: {list(unseen)[:5]}")

    if issues:
        print("⚠ ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ Train/Test consistency OK")

    return issues
```

### Étape 10 : Pipeline de Nettoyage Complet

```python
def full_cleaning_pipeline(train, test, target_col, id_col=None):
    """Pipeline de nettoyage complet, reproductible."""

    # 1. Audit initial
    print("=" * 60)
    print("STEP 1: Audit")
    data_audit(train, 'Train')
    data_audit(test, 'Test')

    # 2. Fixer les types
    print("\n" + "=" * 60)
    print("STEP 2: Fix data types")
    train = fix_data_types(train)
    test = fix_data_types(test)

    # 3. Identifier colonnes
    exclude = [target_col] + ([id_col] if id_col else [])
    num_cols = train.select_dtypes(include=[np.number]).columns.drop(exclude, errors='ignore').tolist()
    cat_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()

    # 4. Doublons
    print("\n" + "=" * 60)
    print("STEP 3: Duplicates")
    train = handle_duplicates(train)

    # 5. Missing values
    print("\n" + "=" * 60)
    print("STEP 4: Missing values")
    missing_analysis = analyze_missing_pattern(train, target_col)
    train, test = smart_impute(train, test, num_cols, cat_cols)

    # 6. Outliers (clip, ne pas supprimer)
    print("\n" + "=" * 60)
    print("STEP 5: Outliers")
    outlier_report = detect_outliers(train, num_cols)
    for col in num_cols:
        if outlier_report.loc[col, 'pct'] > 5:
            train = clip_outliers(train, col)
            test = clip_outliers(test, col)

    # 7. Validation consistency
    print("\n" + "=" * 60)
    print("STEP 6: Train/Test consistency")
    features = [c for c in train.columns if c not in exclude]
    validate_train_test_consistency(train, test, features)

    print("\n" + "=" * 60)
    print(f"DONE: Train {train.shape}, Test {test.shape}")
    return train, test
```

## Règles d'Or du Nettoyage Kaggle

1. **TOUJOURS fit sur train, transform sur test** (scalers, imputers, encoders)
2. **Missing values** : créer un indicateur `_is_null` AVANT d'imputer → souvent informatif
3. **Outliers** : clipper (winsorize) plutôt que supprimer → préserve les données test
4. **Encodage** : vérifier avec `chardet` avant de lire les CSV
5. **Catégories unseen** : gérer les catégories présentes dans test mais pas dans train
6. **Réduction mémoire** : `reduce_mem_usage()` après nettoyage, avant feature engineering
7. **Ne PAS normaliser pour les GBDT** : XGBoost/LightGBM/CatBoost n'en ont pas besoin
8. **Normaliser pour les NN/SVM/KNN** : StandardScaler ou RobustScaler
9. **Log transform** : pour les features avec skewness > 1 et valeurs > 0
10. **Documenter** : chaque décision de nettoyage dans un log/markdown
