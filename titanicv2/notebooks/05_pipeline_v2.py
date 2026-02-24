"""Pipeline V2 - Anti-overfitting Titanic"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import os
import warnings
warnings.filterwarnings('ignore')

SEED = 42
N_FOLDS = 10
np.random.seed(SEED)

# =============================================
# FEATURE ENGINEERING V2 - NO LEAKAGE
# =============================================
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

for df in [train, test]:
    # Title
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_map = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs', 'Countess': 'Rare',
        'Lady': 'Rare', 'Sir': 'Rare', 'Don': 'Rare', 'Dona': 'Rare',
        'Jonkheer': 'Rare', 'Capt': 'Rare'
    }
    df['Title'] = df['Title'].map(title_map).fillna('Rare')

# Age imputation from TRAIN medians only
age_by_title = train.groupby('Title')['Age'].median()
for df in [train, test]:
    for title in df['Title'].unique():
        mask = (df['Age'].isnull()) & (df['Title'] == title)
        if title in age_by_title.index:
            df.loc[mask, 'Age'] = age_by_title[title]
        else:
            df.loc[mask, 'Age'] = train['Age'].median()

# Embarked/Fare from TRAIN
embarked_mode = train['Embarked'].mode()[0]
fare_by_class = train.groupby('Pclass')['Fare'].median()
for df in [train, test]:
    df['Embarked'] = df['Embarked'].fillna(embarked_mode)
    for pclass in [1, 2, 3]:
        mask = (df['Fare'].isnull()) & (df['Pclass'] == pclass)
        df.loc[mask, 'Fare'] = fare_by_class[pclass]

# Build features - NO frequency encodings, NO combined data tricks
for df in [train, test]:
    df['Sex_enc'] = (df['Sex'] == 'male').astype(int)
    title_enc = {'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Rare': 4}
    df['Title_enc'] = df['Title'].map(title_enc).fillna(4).astype(int)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['HasCabin'] = df['Cabin'].notna().astype(int)
    df['IsChild'] = (df['Age'] <= 12).astype(int)
    df['Age_Pclass'] = df['Age'] * df['Pclass']
    df['LogFare'] = np.log1p(df['Fare'])
    emb_enc = {'C': 0, 'Q': 1, 'S': 2}
    df['Embarked_enc'] = df['Embarked'].map(emb_enc).fillna(2).astype(int)

features = [
    'Pclass', 'Sex_enc', 'Age', 'SibSp', 'Parch', 'Fare',
    'Title_enc', 'FamilySize', 'IsAlone',
    'HasCabin', 'IsChild', 'Age_Pclass',
    'LogFare', 'Embarked_enc',
]

X = train[features].values
y = train['Survived'].values.astype(int)
X_test = test[features].values
test_ids = test['PassengerId'].values

print(f"=== PIPELINE V2 - ANTI-OVERFITTING ===")
print(f"X: {X.shape}, y: {y.shape}, X_test: {X_test.shape}")
print(f"Features ({len(features)}): {features}")
print(f"NaN: train={np.isnan(X).sum()}, test={np.isnan(X_test).sum()}")

# =============================================
# MODELING - CONSERVATIVE PARAMS, 10-FOLD
# =============================================
kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

results = {}

# --- Logistic Regression ---
print(f"\n=== MODELS (10-Fold CV, conservative) ===\n")
oof_lr = np.zeros(len(y)); test_lr = np.zeros(len(test_ids)); scores_lr = []
for ti, vi in kf.split(X_scaled, y):
    m = LogisticRegression(C=0.5, max_iter=1000, random_state=SEED)
    m.fit(X_scaled[ti], y[ti])
    oof_lr[vi] = m.predict_proba(X_scaled[vi])[:, 1]
    test_lr += m.predict_proba(X_test_scaled)[:, 1] / N_FOLDS
    scores_lr.append(accuracy_score(y[vi], m.predict(X_scaled[vi])))
print(f"LR:  {np.mean(scores_lr):.5f} +/- {np.std(scores_lr):.5f}")
results['LR'] = {'oof': oof_lr, 'test': test_lr, 'acc': np.mean(scores_lr)}

# --- Random Forest ---
oof_rf = np.zeros(len(y)); test_rf = np.zeros(len(test_ids)); scores_rf = []
for ti, vi in kf.split(X, y):
    m = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_split=10,
        min_samples_leaf=5, max_features='sqrt', random_state=SEED, n_jobs=-1)
    m.fit(X[ti], y[ti])
    oof_rf[vi] = m.predict_proba(X[vi])[:, 1]
    test_rf += m.predict_proba(X_test)[:, 1] / N_FOLDS
    scores_rf.append(accuracy_score(y[vi], m.predict(X[vi])))
print(f"RF:  {np.mean(scores_rf):.5f} +/- {np.std(scores_rf):.5f}")
results['RF'] = {'oof': oof_rf, 'test': test_rf, 'acc': np.mean(scores_rf)}

# --- SVM ---
oof_svm = np.zeros(len(y)); test_svm = np.zeros(len(test_ids)); scores_svm = []
for ti, vi in kf.split(X_scaled, y):
    m = SVC(C=0.8, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
    m.fit(X_scaled[ti], y[ti])
    oof_svm[vi] = m.predict_proba(X_scaled[vi])[:, 1]
    test_svm += m.predict_proba(X_test_scaled)[:, 1] / N_FOLDS
    scores_svm.append(accuracy_score(y[vi], m.predict(X_scaled[vi])))
print(f"SVM: {np.mean(scores_svm):.5f} +/- {np.std(scores_svm):.5f}")
results['SVM'] = {'oof': oof_svm, 'test': test_svm, 'acc': np.mean(scores_svm)}

# --- LightGBM conservative ---
lgb_params = {
    'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
    'num_leaves': 15, 'learning_rate': 0.05, 'feature_fraction': 0.7,
    'bagging_fraction': 0.7, 'bagging_freq': 5, 'min_child_samples': 30,
    'reg_alpha': 1.0, 'reg_lambda': 5.0, 'n_estimators': 300,
    'verbose': -1, 'random_state': SEED,
}
oof_lgb = np.zeros(len(y)); test_lgb = np.zeros(len(test_ids)); scores_lgb = []
for ti, vi in kf.split(X, y):
    m = lgb.LGBMClassifier(**lgb_params)
    m.fit(X[ti], y[ti], eval_set=[(X[vi], y[vi])],
          callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
    oof_lgb[vi] = m.predict_proba(X[vi])[:, 1]
    test_lgb += m.predict_proba(X_test)[:, 1] / N_FOLDS
    scores_lgb.append(accuracy_score(y[vi], m.predict(X[vi])))
print(f"LGB: {np.mean(scores_lgb):.5f} +/- {np.std(scores_lgb):.5f}")
results['LGB'] = {'oof': oof_lgb, 'test': test_lgb, 'acc': np.mean(scores_lgb)}

# --- XGBoost conservative ---
xgb_params = {
    'objective': 'binary:logistic', 'eval_metric': 'logloss',
    'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.7,
    'colsample_bytree': 0.7, 'min_child_weight': 10,
    'reg_alpha': 1.0, 'reg_lambda': 5.0, 'gamma': 0.5,
    'n_estimators': 300, 'random_state': SEED, 'verbosity': 0,
}
oof_xgb = np.zeros(len(y)); test_xgb = np.zeros(len(test_ids)); scores_xgb = []
for ti, vi in kf.split(X, y):
    m = xgb.XGBClassifier(**xgb_params)
    m.fit(X[ti], y[ti], eval_set=[(X[vi], y[vi])], verbose=False)
    oof_xgb[vi] = m.predict_proba(X[vi])[:, 1]
    test_xgb += m.predict_proba(X_test)[:, 1] / N_FOLDS
    scores_xgb.append(accuracy_score(y[vi], m.predict(X[vi])))
print(f"XGB: {np.mean(scores_xgb):.5f} +/- {np.std(scores_xgb):.5f}")
results['XGB'] = {'oof': oof_xgb, 'test': test_xgb, 'acc': np.mean(scores_xgb)}

# --- CatBoost conservative ---
oof_cb = np.zeros(len(y)); test_cb = np.zeros(len(test_ids)); scores_cb = []
for ti, vi in kf.split(X, y):
    m = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05,
        l2_leaf_reg=5.0, random_seed=SEED, verbose=0, early_stopping_rounds=30)
    m.fit(X[ti], y[ti], eval_set=(X[vi], y[vi]), verbose=0)
    oof_cb[vi] = m.predict_proba(X[vi])[:, 1]
    test_cb += m.predict_proba(X_test)[:, 1] / N_FOLDS
    scores_cb.append(accuracy_score(y[vi], m.predict(X[vi])))
print(f"CB:  {np.mean(scores_cb):.5f} +/- {np.std(scores_cb):.5f}")
results['CB'] = {'oof': oof_cb, 'test': test_cb, 'acc': np.mean(scores_cb)}

# =============================================
# ENSEMBLES - SIMPLE, NO WEIGHT OPTIMIZATION
# =============================================
print(f"\n=== ENSEMBLES V2 ===\n")

all_names = list(results.keys())
all_oof = np.column_stack([results[m]['oof'] for m in all_names])
all_test = np.column_stack([results[m]['test'] for m in all_names])

# Simple average all
avg_acc = accuracy_score(y, (all_oof.mean(axis=1) > 0.5).astype(int))
print(f"Simple Average (all 6):     {avg_acc:.5f}")

# Top-3 GBDT
top3 = ['LGB', 'XGB', 'CB']
top3_oof = np.column_stack([results[m]['oof'] for m in top3]).mean(axis=1)
top3_test = np.column_stack([results[m]['test'] for m in top3]).mean(axis=1)
top3_acc = accuracy_score(y, (top3_oof > 0.5).astype(int))
print(f"Top-3 GBDT (LGB+XGB+CB):   {top3_acc:.5f}")

# Diverse: LR + RF + CB
diverse = ['LR', 'RF', 'CB']
div_oof = np.column_stack([results[m]['oof'] for m in diverse]).mean(axis=1)
div_test = np.column_stack([results[m]['test'] for m in diverse]).mean(axis=1)
div_acc = accuracy_score(y, (div_oof > 0.5).astype(int))
print(f"Diverse (LR+RF+CB):        {div_acc:.5f}")

# Majority vote
votes_oof = (all_oof > 0.5).astype(int)
votes_test = (all_test > 0.5).astype(int)
maj_oof = (votes_oof.mean(axis=1) > 0.5).astype(int)
maj_test = (votes_test.mean(axis=1) > 0.5).astype(int)
maj_acc = accuracy_score(y, maj_oof)
print(f"Majority Vote (all 6):      {maj_acc:.5f}")

# =============================================
# SUBMISSIONS
# =============================================
os.makedirs('../submissions', exist_ok=True)

subs = {
    'v2_avg_all': (all_test.mean(axis=1) > 0.5).astype(int),
    'v2_top3_gbdt': (top3_test > 0.5).astype(int),
    'v2_diverse': (div_test > 0.5).astype(int),
    'v2_majority': maj_test,
    'v2_lr': (results['LR']['test'] > 0.5).astype(int),
    'v2_svm': (results['SVM']['test'] > 0.5).astype(int),
    'v2_cb': (results['CB']['test'] > 0.5).astype(int),
}

print(f"\n=== SUBMISSIONS V2 ===")
for name, preds in subs.items():
    sub = pd.DataFrame({'PassengerId': test_ids.astype(int), 'Survived': preds.astype(int)})
    sub.to_csv(f'../submissions/{name}.csv', index=False)
    print(f"  {name}: survival={sub.Survived.mean():.3f}")

print(f"\n{len(subs)} submissions. Recommandation: soumettre v2_avg_all et v2_lr")
