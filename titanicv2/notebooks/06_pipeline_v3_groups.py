"""Pipeline V3 - Group Survival Trick + Conservative CatBoost
Based on V2 anti-overfitting pipeline + group survival features.
"""
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
# 1. LOAD + BASE FEATURES (same as V2)
# =============================================
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

for df in [train, test]:
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_map = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs', 'Countess': 'Rare',
        'Lady': 'Rare', 'Sir': 'Rare', 'Don': 'Rare', 'Dona': 'Rare',
        'Jonkheer': 'Rare', 'Capt': 'Rare'
    }
    df['Title'] = df['Title'].map(title_map).fillna('Rare')

age_by_title = train.groupby('Title')['Age'].median()
for df in [train, test]:
    for title in df['Title'].unique():
        mask = (df['Age'].isnull()) & (df['Title'] == title)
        if title in age_by_title.index:
            df.loc[mask, 'Age'] = age_by_title[title]
        else:
            df.loc[mask, 'Age'] = train['Age'].median()

embarked_mode = train['Embarked'].mode()[0]
fare_by_class = train.groupby('Pclass')['Fare'].median()
for df in [train, test]:
    df['Embarked'] = df['Embarked'].fillna(embarked_mode)
    for pclass in [1, 2, 3]:
        mask = (df['Fare'].isnull()) & (df['Pclass'] == pclass)
        df.loc[mask, 'Fare'] = fare_by_class[pclass]

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
    df['Surname'] = df['Name'].str.split(',').str[0]

# =============================================
# 2. GROUP SURVIVAL TRICK
# =============================================
print("=== GROUP SURVIVAL FEATURES ===\n")

# --- 2a. Ticket Group Survival Rate ---
# Compute survival stats per ticket group using TRAIN ONLY
ticket_surv = train.groupby('Ticket')['Survived'].agg(['mean', 'count'])
ticket_surv.columns = ['TicketGroupSurvRate', 'TicketGroupSize_train']

# For train passengers: use leave-one-out to avoid leaking own label
# surv_rate_others = (group_sum - own_label) / (group_count - 1)
train_ticket_sum = train.groupby('Ticket')['Survived'].transform('sum')
train_ticket_cnt = train.groupby('Ticket')['Survived'].transform('count')

train['TicketGroupSurvRate'] = np.where(
    train_ticket_cnt > 1,
    (train_ticket_sum - train['Survived']) / (train_ticket_cnt - 1),
    -1  # sentinel for solo ticket holders
)

# For test: use full train group mean (no leakage - test labels not used)
test['TicketGroupSurvRate'] = test['Ticket'].map(
    ticket_surv['TicketGroupSurvRate']
).fillna(-1)  # -1 for tickets not seen in train

# Ticket group size from combined (this is structural, not label leakage)
combined_ticket_size = pd.concat([train['Ticket'], test['Ticket']]).value_counts()
for df in [train, test]:
    df['TicketGroupSize'] = df['Ticket'].map(combined_ticket_size)

n_train_with_group = (train['TicketGroupSurvRate'] != -1).sum()
n_test_with_group = (test['TicketGroupSurvRate'] != -1).sum()
print(f"Ticket Group Survival Rate:")
print(f"  Train passengers with group info: {n_train_with_group}/{len(train)}")
print(f"  Test passengers with group info:  {n_test_with_group}/{len(test)}")

# --- 2b. Family Group Survival Rate ---
# Family group = Surname + FamilySize (same last name + same family size = same family)
for df in [train, test]:
    df['FamilyGroup'] = df['Surname'] + '_' + df['FamilySize'].astype(str)

# Compute from train only, leave-one-out for train
family_surv = train.groupby('FamilyGroup')['Survived'].agg(['mean', 'sum', 'count'])

train_fg_sum = train.groupby('FamilyGroup')['Survived'].transform('sum')
train_fg_cnt = train.groupby('FamilyGroup')['Survived'].transform('count')

train['FamilyGroupSurvRate'] = np.where(
    train_fg_cnt > 1,
    (train_fg_sum - train['Survived']) / (train_fg_cnt - 1),
    -1
)

test['FamilyGroupSurvRate'] = test['FamilyGroup'].map(
    family_surv['mean']
).fillna(-1)

n_train_fam = (train['FamilyGroupSurvRate'] != -1).sum()
n_test_fam = (test['FamilyGroupSurvRate'] != -1).sum()
print(f"\nFamily Group Survival Rate:")
print(f"  Train passengers with family info: {n_train_fam}/{len(train)}")
print(f"  Test passengers with family info:  {n_test_fam}/{len(test)}")

# --- 2c. Sex-specific group survival ---
# Women/children in groups where women/children survived -> strong signal
# Men in groups where men survived -> moderate signal

# WomenChildren flag
for df in [train, test]:
    df['IsWomanChild'] = ((df['Sex'] == 'female') | (df['Age'] <= 12)).astype(int)

# Per ticket: survival rate of women/children from train
wc_train = train[train['IsWomanChild'] == 1]
men_train = train[train['IsWomanChild'] == 0]

wc_ticket_surv = wc_train.groupby('Ticket')['Survived'].mean()
men_ticket_surv = men_train.groupby('Ticket')['Survived'].mean()

for df in [train, test]:
    df['TicketWCSurvRate'] = df['Ticket'].map(wc_ticket_surv).fillna(-1)
    df['TicketMenSurvRate'] = df['Ticket'].map(men_ticket_surv).fillna(-1)

# For train WC: leave-one-out on WC subgroup
for idx in train.index:
    ticket = train.loc[idx, 'Ticket']
    is_wc = train.loc[idx, 'IsWomanChild']
    surv = train.loc[idx, 'Survived']

    if is_wc == 1:
        group = wc_train[(wc_train['Ticket'] == ticket) & (wc_train.index != idx)]
        if len(group) > 0:
            train.loc[idx, 'TicketWCSurvRate'] = group['Survived'].mean()
        else:
            train.loc[idx, 'TicketWCSurvRate'] = -1
    else:
        group = men_train[(men_train['Ticket'] == ticket) & (men_train.index != idx)]
        if len(group) > 0:
            train.loc[idx, 'TicketMenSurvRate'] = group['Survived'].mean()
        else:
            train.loc[idx, 'TicketMenSurvRate'] = -1

# --- 2d. Combined group signal ---
# If you are a woman and your ticket group women survived -> you likely survived
# If you are a man and your ticket group men died -> you likely died
for df in [train, test]:
    # Best available group rate for this person
    df['GroupSurvSignal'] = np.where(
        df['IsWomanChild'] == 1,
        np.where(df['TicketWCSurvRate'] >= 0, df['TicketWCSurvRate'],
                 np.where(df['FamilyGroupSurvRate'] >= 0, df['FamilyGroupSurvRate'],
                          np.where(df['TicketGroupSurvRate'] >= 0, df['TicketGroupSurvRate'], -1))),
        np.where(df['TicketMenSurvRate'] >= 0, df['TicketMenSurvRate'],
                 np.where(df['FamilyGroupSurvRate'] >= 0, df['FamilyGroupSurvRate'],
                          np.where(df['TicketGroupSurvRate'] >= 0, df['TicketGroupSurvRate'], -1)))
    )

    # Has any group info
    df['HasGroupInfo'] = (df['GroupSurvSignal'] >= 0).astype(int)

print(f"\nCombined Group Signal:")
print(f"  Train with signal: {(train['HasGroupInfo']==1).sum()}/{len(train)}")
print(f"  Test with signal:  {(test['HasGroupInfo']==1).sum()}/{len(test)}")

# =============================================
# 3. FEATURE SELECTION
# =============================================
features_v2 = [
    'Pclass', 'Sex_enc', 'Age', 'SibSp', 'Parch', 'Fare',
    'Title_enc', 'FamilySize', 'IsAlone',
    'HasCabin', 'IsChild', 'Age_Pclass',
    'LogFare', 'Embarked_enc',
]

group_features = [
    'TicketGroupSurvRate',
    'TicketGroupSize',
    'FamilyGroupSurvRate',
    'TicketWCSurvRate',
    'TicketMenSurvRate',
    'GroupSurvSignal',
    'HasGroupInfo',
    'IsWomanChild',
]

features_v3 = features_v2 + group_features

X = train[features_v3].values
y = train['Survived'].values.astype(int)
X_test = test[features_v3].values
test_ids = test['PassengerId'].values

print(f"\n=== PIPELINE V3 ===")
print(f"X: {X.shape}, y: {y.shape}, X_test: {X_test.shape}")
print(f"V2 features: {len(features_v2)}, Group features: {len(group_features)}, Total: {len(features_v3)}")
print(f"NaN: train={np.isnan(X).sum()}, test={np.isnan(X_test).sum()}")

# =============================================
# 4. MODELING - SAME CONSERVATIVE PARAMS AS V2
# =============================================
kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

results = {}

print(f"\n=== MODELS V3 (10-Fold CV, conservative) ===\n")

# --- Logistic Regression ---
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
# 5. FEATURE IMPORTANCE
# =============================================
print(f"\n=== FEATURE IMPORTANCE (CatBoost) ===\n")
cb_full = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05,
    l2_leaf_reg=5.0, random_seed=SEED, verbose=0)
cb_full.fit(X, y)
imp = pd.DataFrame({'feature': features_v3, 'importance': cb_full.feature_importances_})
imp = imp.sort_values('importance', ascending=False)
for _, row in imp.iterrows():
    marker = " <-- GROUP" if row['feature'] in group_features else ""
    print(f"  {row['importance']:6.1f}  {row['feature']}{marker}")

# =============================================
# 6. ENSEMBLES
# =============================================
print(f"\n=== ENSEMBLES V3 ===\n")

all_names = list(results.keys())
all_oof = np.column_stack([results[m]['oof'] for m in all_names])
all_test = np.column_stack([results[m]['test'] for m in all_names])

avg_acc = accuracy_score(y, (all_oof.mean(axis=1) > 0.5).astype(int))
print(f"Simple Average (all 6):     {avg_acc:.5f}")

top3 = ['LGB', 'XGB', 'CB']
top3_test = np.column_stack([results[m]['test'] for m in top3]).mean(axis=1)
top3_oof = np.column_stack([results[m]['oof'] for m in top3]).mean(axis=1)
top3_acc = accuracy_score(y, (top3_oof > 0.5).astype(int))
print(f"Top-3 GBDT (LGB+XGB+CB):   {top3_acc:.5f}")

votes_test = (all_test > 0.5).astype(int)
votes_oof = (all_oof > 0.5).astype(int)
maj_test = (votes_test.mean(axis=1) > 0.5).astype(int)
maj_oof = (votes_oof.mean(axis=1) > 0.5).astype(int)
maj_acc = accuracy_score(y, maj_oof)
print(f"Majority Vote (all 6):      {maj_acc:.5f}")

# =============================================
# 7. SUBMISSIONS
# =============================================
os.makedirs('../submissions', exist_ok=True)

subs = {
    'v3_avg_all': (all_test.mean(axis=1) > 0.5).astype(int),
    'v3_top3_gbdt': (top3_test > 0.5).astype(int),
    'v3_majority': maj_test,
    'v3_cb': (results['CB']['test'] > 0.5).astype(int),
    'v3_lgb': (results['LGB']['test'] > 0.5).astype(int),
    'v3_xgb': (results['XGB']['test'] > 0.5).astype(int),
    'v3_svm': (results['SVM']['test'] > 0.5).astype(int),
}

print(f"\n=== SUBMISSIONS V3 ===")
for name, preds in subs.items():
    sub = pd.DataFrame({'PassengerId': test_ids.astype(int), 'Survived': preds.astype(int)})
    sub.to_csv(f'../submissions/{name}.csv', index=False)
    print(f"  {name}: survival={sub.Survived.mean():.3f}")

print(f"\n{len(subs)} submissions created.")
print(f"Recommandation: v3_cb et v3_avg_all en priorite")
