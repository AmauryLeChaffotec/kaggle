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

### Étape 10b : NaN Déguisés

```python
def fix_disguised_nans(df, extra_markers=None):
    """Détecte et remplace les NaN déguisés en vrais np.nan.
    Les datasets Kaggle contiennent souvent des marqueurs de missing
    qui ne sont PAS détectés par pandas automatiquement.
    """
    # Marqueurs courants de NaN
    nan_markers = [
        'N/A', 'n/a', 'NA', 'na', 'N.A.', 'n.a.',
        'None', 'none', 'NONE',
        'null', 'NULL', 'Null',
        'NaN', 'nan',
        'missing', 'Missing', 'MISSING',
        '-', '--', '---',
        '?', '??',
        '.', '...',
        '', ' ', '  ',
        'not available', 'Not Available',
        'unknown', 'Unknown', 'UNKNOWN',
        'undefined', 'Undefined',
        '-1', '-999', '-9999', '999', '9999',
        '#N/A', '#NA', '#NULL!', '#REF!',
        'inf', '-inf', 'Inf', '-Inf',
    ]

    if extra_markers:
        nan_markers.extend(extra_markers)

    n_fixed_total = 0
    for col in df.select_dtypes(include=['object']).columns:
        # Strip whitespace d'abord
        df[col] = df[col].str.strip()

        # Compter les remplacements
        mask = df[col].isin(nan_markers)
        n_fixed = mask.sum()
        if n_fixed > 0:
            df.loc[mask, col] = np.nan
            n_fixed_total += n_fixed
            print(f"  {col}: {n_fixed} disguised NaN found and replaced")

    # Vérifier aussi les colonnes numériques pour les sentinelles (-999, 9999, etc.)
    for col in df.select_dtypes(include=[np.number]).columns:
        for sentinel in [-999, -9999, 9999, -1]:
            n = (df[col] == sentinel).sum()
            if n > 0 and n < len(df) * 0.5:  # Pas plus de 50%, sinon c'est une vraie valeur
                pct = n / len(df) * 100
                if pct > 0.1:  # Au moins 0.1% pour être significatif
                    print(f"  ⚠ {col}: {n} values == {sentinel} ({pct:.1f}%) — possible sentinel NaN")

    print(f"Total disguised NaN fixed: {n_fixed_total}")
    return df
```

### Étape 10c : Rare Categories

```python
def handle_rare_categories(train_df, test_df, cat_cols, min_count=10,
                           min_pct=0.01, replace_with='_RARE_'):
    """Regroupe les catégories rares en une seule catégorie.

    Pourquoi : les catégories avec <10 occurrences causent du bruit,
    de l'overfitting, et des problèmes avec le target encoding.

    min_count : nombre minimal d'occurrences dans le TRAIN
    min_pct : pourcentage minimal d'occurrences dans le TRAIN
    """
    for col in cat_cols:
        # Compter les occurrences dans le train
        value_counts = train_df[col].value_counts()
        n_total = len(train_df)
        min_threshold = max(min_count, int(n_total * min_pct))

        # Identifier les catégories rares
        rare_cats = value_counts[value_counts < min_threshold].index.tolist()

        if rare_cats:
            n_rare = len(rare_cats)
            n_affected = train_df[col].isin(rare_cats).sum()
            print(f"  {col}: {n_rare} rare categories ({n_affected} rows) → '{replace_with}'")

            # Remplacer dans train ET test
            train_df[col] = train_df[col].replace(rare_cats, replace_with)
            test_df[col] = test_df[col].replace(rare_cats, replace_with)

            # Les catégories unseen dans le test deviennent aussi _RARE_
            train_cats = set(train_df[col].dropna().unique())
            test_only = set(test_df[col].dropna().unique()) - train_cats
            if test_only:
                print(f"    + {len(test_only)} unseen test categories → '{replace_with}'")
                test_df[col] = test_df[col].replace(list(test_only), replace_with)

    return train_df, test_df
```

### Étape 10d : Colonnes Constantes et Quasi-Constantes

```python
def remove_constant_columns(train_df, test_df, quasi_threshold=0.999):
    """Supprime les colonnes constantes ou quasi-constantes.

    quasi_threshold=0.999 : supprime si 99.9%+ des valeurs sont identiques.
    Ces colonnes n'apportent aucune information au modèle.
    """
    cols_to_drop = []

    for col in train_df.columns:
        n_unique = train_df[col].nunique(dropna=False)

        # Constante (1 seule valeur, ou 1 valeur + NaN)
        if n_unique <= 1:
            cols_to_drop.append(col)
            print(f"  DROP {col}: constant (1 unique value)")
            continue

        # Quasi-constante
        if train_df[col].dtype in ['object', 'category']:
            top_pct = train_df[col].value_counts(normalize=True, dropna=False).iloc[0]
        else:
            top_pct = train_df[col].value_counts(normalize=True, dropna=False).iloc[0]

        if top_pct >= quasi_threshold:
            cols_to_drop.append(col)
            print(f"  DROP {col}: quasi-constant ({top_pct*100:.1f}% same value)")

    if cols_to_drop:
        train_df = train_df.drop(columns=cols_to_drop)
        test_df = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns])
        print(f"\nRemoved {len(cols_to_drop)} constant/quasi-constant columns")
    else:
        print("No constant columns found")

    return train_df, test_df
```

### Étape 10e : High Cardinality Detection

```python
def detect_high_cardinality(train_df, test_df, target_col=None,
                            high_threshold=50, strategies=None):
    """Détecte les colonnes catégorielles à haute cardinalité et propose
    une stratégie pour chacune.

    high_threshold : au-delà de 50 catégories uniques = high cardinality
    """
    cat_cols = train_df.select_dtypes(include=['object', 'category']).columns
    report = {}

    print("=" * 60)
    print("HIGH CARDINALITY ANALYSIS")
    print("=" * 60)

    for col in cat_cols:
        n_unique_train = train_df[col].nunique()
        n_unique_test = test_df[col].nunique() if col in test_df.columns else 0
        unseen = 0
        if col in test_df.columns:
            unseen = len(set(test_df[col].dropna().unique()) - set(train_df[col].dropna().unique()))

        if n_unique_train >= high_threshold:
            # Calculer la corrélation avec le target via frequency encoding
            target_corr = 0
            if target_col and target_col in train_df.columns:
                freq = train_df[col].map(train_df[col].value_counts())
                target_corr = abs(freq.corr(train_df[target_col]))

            strategy = _suggest_cardinality_strategy(n_unique_train, unseen, target_corr)

            report[col] = {
                'n_unique_train': n_unique_train,
                'n_unique_test': n_unique_test,
                'unseen_in_test': unseen,
                'target_corr_freq': round(target_corr, 4),
                'strategy': strategy
            }

            print(f"\n  [{col}] — {n_unique_train} categories")
            print(f"    Unseen in test: {unseen}")
            print(f"    Target corr (freq): {target_corr:.4f}")
            print(f"    → Strategy: {strategy}")

    if not report:
        print("No high cardinality columns found")

    return report

def _suggest_cardinality_strategy(n_unique, unseen, target_corr):
    """Suggère la stratégie optimale pour une colonne high-cardinality."""
    if n_unique > 10000:
        return "HASH encoding (trop de catégories pour target encoding)"
    elif unseen > n_unique * 0.3:
        return "FREQUENCY encoding (30%+ unseen dans test, target encoding risqué)"
    elif target_corr > 0.1:
        return "TARGET encoding (Bayesian smoothing, m=20, 10-fold OOF)"
    elif n_unique > 200:
        return "FREQUENCY + RARE grouping (groupe <10 en _RARE_, puis frequency)"
    else:
        return "ORDINAL ou TARGET encoding (cardinalité modérée)"
```

### Étape 10f : Multi-Valued Cells

```python
def handle_multi_valued_cells(df, col, separator=',', max_categories=20,
                              method='onehot'):
    """Parse les cellules contenant plusieurs valeurs séparées par un délimiteur.

    Exemple : "Action,Comedy,Drama" → 3 colonnes binaires
    Exemple : "B/0/P" → 3 colonnes séparées (deck, num, side)

    method='onehot' : crée une colonne binaire par valeur unique
    method='split' : split en N colonnes séparées (pour structure fixe)
    method='count' : juste compter le nombre de valeurs
    """
    if method == 'onehot':
        # Extraire toutes les valeurs uniques
        all_values = set()
        df[col].dropna().str.split(separator).apply(
            lambda x: all_values.update([v.strip() for v in x])
        )

        # Limiter aux top N catégories
        if len(all_values) > max_categories:
            # Garder les plus fréquentes
            from collections import Counter
            counter = Counter()
            df[col].dropna().str.split(separator).apply(
                lambda x: counter.update([v.strip() for v in x])
            )
            all_values = {v for v, _ in counter.most_common(max_categories)}
            print(f"  {col}: {len(counter)} values → keeping top {max_categories}")

        # Créer les colonnes one-hot
        for val in sorted(all_values):
            safe_name = f"{col}_{val.replace(' ', '_')}"
            df[safe_name] = df[col].fillna('').str.contains(
                re.escape(val.strip()), case=False
            ).astype(int)

        print(f"  {col}: created {len(all_values)} binary columns")

    elif method == 'split':
        # Split en colonnes séparées
        split_df = df[col].str.split(separator, expand=True)
        for i in range(split_df.shape[1]):
            df[f"{col}_part{i}"] = split_df[i].str.strip()
        print(f"  {col}: split into {split_df.shape[1]} columns")

    elif method == 'count':
        # Juste compter le nombre de valeurs
        df[f"{col}_count"] = df[col].fillna('').str.split(separator).apply(len)
        df.loc[df[col].isna(), f"{col}_count"] = 0
        print(f"  {col}: created count column")

    return df
```

## Pipeline de Nettoyage Complet (mise à jour)

```python
def full_cleaning_pipeline_v2(train, test, target_col, id_col=None,
                               cat_cols=None, extra_nan_markers=None):
    """Pipeline de nettoyage complet V2 — couvre TOUS les cas.

    Étapes :
    1. Audit initial
    2. NaN déguisés → vrais NaN
    3. Fix types
    4. Doublons
    5. Colonnes constantes/quasi-constantes
    6. Missing values (analyse + imputation)
    7. Outliers (détection + winsorization)
    8. Rare categories (groupement)
    9. High cardinality (détection + stratégie)
    10. Multi-valued cells (si détectés)
    11. Train/Test consistency
    """
    exclude = [target_col] + ([id_col] if id_col else [])

    # 1. Audit
    print("=" * 60)
    print("STEP 1: Audit")
    data_audit(train, 'Train')
    data_audit(test, 'Test')

    # 2. NaN déguisés
    print("\n" + "=" * 60)
    print("STEP 2: Fix disguised NaN")
    train = fix_disguised_nans(train, extra_nan_markers)
    test = fix_disguised_nans(test, extra_nan_markers)

    # 3. Fix types
    print("\n" + "=" * 60)
    print("STEP 3: Fix data types")
    train = fix_data_types(train)
    test = fix_data_types(test)

    # 4. Identifier colonnes
    num_cols = train.select_dtypes(include=[np.number]).columns.drop(exclude, errors='ignore').tolist()
    if cat_cols is None:
        cat_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()

    # 5. Doublons
    print("\n" + "=" * 60)
    print("STEP 4: Duplicates")
    train = handle_duplicates(train)

    # 6. Colonnes constantes
    print("\n" + "=" * 60)
    print("STEP 5: Constant/quasi-constant columns")
    train, test = remove_constant_columns(train, test)
    # Refresh col lists
    num_cols = [c for c in num_cols if c in train.columns]
    cat_cols = [c for c in cat_cols if c in train.columns]

    # 7. Missing values
    print("\n" + "=" * 60)
    print("STEP 6: Missing values")
    missing_analysis = analyze_missing_pattern(train, target_col)
    train, test = smart_impute(train, test, num_cols, cat_cols)

    # 8. Outliers
    print("\n" + "=" * 60)
    print("STEP 7: Outliers")
    outlier_report = detect_outliers(train, num_cols)
    for col in num_cols:
        if col in outlier_report.index and outlier_report.loc[col, 'pct'] > 5:
            train = clip_outliers(train, col)
            test = clip_outliers(test, col)

    # 9. Rare categories
    print("\n" + "=" * 60)
    print("STEP 8: Rare categories")
    train, test = handle_rare_categories(train, test, cat_cols)

    # 10. High cardinality
    print("\n" + "=" * 60)
    print("STEP 9: High cardinality analysis")
    hc_report = detect_high_cardinality(train, test, target_col)

    # 11. Consistency
    print("\n" + "=" * 60)
    print("STEP 10: Train/Test consistency")
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

## Rapport de Sortie (OBLIGATOIRE)

À la fin du nettoyage, TOUJOURS sauvegarder :
1. Rapport dans : `reports/cleaning/YYYY-MM-DD_cleaning.md` (missing traités, outliers, types corrigés, résumé)
2. Données nettoyées dans : `data/train_clean.parquet` et `data/test_clean.parquet`
3. Confirmer à l'utilisateur les chemins sauvegardés
