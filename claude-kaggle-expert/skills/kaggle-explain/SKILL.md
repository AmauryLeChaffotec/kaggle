---
name: kaggle-explain
description: Expert en ML Explainability pour compétitions Kaggle. Utiliser quand l'utilisateur veut interpréter un modèle, comprendre les prédictions, calculer l'importance des features avec SHAP, LIME, permutation importance, ou Partial Dependence Plots.
argument-hint: <technique ou modèle à expliquer>
---

# Expert ML Explainability - Kaggle Gold Medal

Tu es un expert en interprétabilité des modèles ML. Tu maîtrises SHAP, LIME, Permutation Importance, Partial Dependence Plots, et toutes les techniques modernes d'explainability. Ces techniques sont cruciales pour le feature engineering guidé et le debugging de modèles — deux clés des solutions gagnantes.

## Pourquoi l'Explainability est Cruciale en Compétition

1. **Debugging** : identifier le data leakage, les features corrompues, les patterns irréalistes
2. **Feature engineering guidé** : comprendre quelles features créer basé sur les interactions détectées
3. **Sélection de features** : supprimer le bruit, garder le signal
4. **Validation** : vérifier que le modèle apprend des patterns sensés
5. **Confiance** : solutions gagnantes qui s'expliquent sont mieux classées dans les write-ups

## 1. Permutation Importance

Mesure la baisse de performance quand on mélange aléatoirement les valeurs d'une feature.

### Avec eli5

```python
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Entraîner le modèle
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Calculer la permutation importance
perm = PermutationImportance(model, random_state=42).fit(X_val, y_val)

# Afficher les résultats
eli5.show_weights(perm, feature_names=X_val.columns.tolist())

# Format DataFrame pour manipulation
perm_df = pd.DataFrame({
    'feature': X_val.columns,
    'importance_mean': perm.feature_importances_,
    'importance_std': perm.feature_importances_std_
}).sort_values('importance_mean', ascending=False)
```

### Avec sklearn (natif)

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    model, X_val, y_val,
    n_repeats=10,
    random_state=42,
    n_jobs=-1,
    scoring='roc_auc'  # ADAPTER à la métrique de la compétition
)

perm_df = pd.DataFrame({
    'feature': X_val.columns,
    'importance_mean': result.importances_mean,
    'importance_std': result.importances_std
}).sort_values('importance_mean', ascending=False)

# Visualisation
fig, ax = plt.subplots(figsize=(10, max(6, len(perm_df.head(30)) * 0.3)))
top = perm_df.head(30).sort_values('importance_mean')
ax.barh(top['feature'], top['importance_mean'], xerr=top['importance_std'],
        color='steelblue', alpha=0.8)
ax.set_title('Permutation Importance (Top 30)')
ax.set_xlabel('Mean Importance')
plt.tight_layout()
plt.show()
```

### Interprétation
- **Valeurs positives élevées** : feature très importante
- **Valeurs proches de 0** : feature peu utile — candidate à la suppression
- **Valeurs négatives** : feature bruitée qui dégrade le modèle — supprimer !
- **Écart-type élevé** : importance instable — besoin de plus de données ou repeats

### Usage Compétition
```python
# Sélection de features basée sur la permutation importance
threshold = 0.001
important_features = perm_df[perm_df['importance_mean'] > threshold]['feature'].tolist()
print(f"Features gardées : {len(important_features)}/{len(X_val.columns)}")

# Entraîner le modèle avec les features sélectionnées
model_selected = RandomForestClassifier(n_estimators=100, random_state=42)
model_selected.fit(X_train[important_features], y_train)
score_all = model.score(X_val, y_val)
score_selected = model_selected.score(X_val[important_features], y_val)
print(f"Score toutes features : {score_all:.6f}")
print(f"Score features sélectionnées : {score_selected:.6f}")
```

## 2. Partial Dependence Plots (PDP)

Montre l'effet marginal moyen d'une feature sur la prédiction du modèle.

### PDP 1D (une feature)

```python
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

# PDP pour une seule feature
fig, ax = plt.subplots(figsize=(10, 6))
PartialDependenceDisplay.from_estimator(
    model, X_val,
    features=['feature_name'],
    kind='average',  # 'average' = PDP classique, 'individual' = ICE, 'both' = les deux
    ax=ax
)
ax.set_title("Partial Dependence Plot : feature_name")
plt.tight_layout()
plt.show()

# PDP pour plusieurs features en une grille
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
features_to_plot = ['feat1', 'feat2', 'feat3', 'feat4', 'feat5', 'feat6']
PartialDependenceDisplay.from_estimator(
    model, X_val,
    features=features_to_plot,
    kind='both',  # PDP moyen + lignes individuelles (ICE)
    ax=axes,
    n_cols=3
)
plt.suptitle("Partial Dependence Plots", fontsize=14)
plt.tight_layout()
plt.show()
```

### PDP 2D (interactions entre 2 features)

```python
# PDP 2D : heatmap d'interaction
fig, ax = plt.subplots(figsize=(10, 8))
PartialDependenceDisplay.from_estimator(
    model, X_val,
    features=[('feature_1', 'feature_2')],
    kind='average',
    ax=ax
)
ax.set_title("Interaction : feature_1 x feature_2")
plt.tight_layout()
plt.show()
```

### ICE Plots (Individual Conditional Expectation)

```python
# ICE = PDP par observation individuelle
fig, ax = plt.subplots(figsize=(10, 6))
PartialDependenceDisplay.from_estimator(
    model, X_val,
    features=['feature_name'],
    kind='individual',  # Chaque ligne = une observation
    subsample=50,       # Limiter pour lisibilité
    ax=ax
)
ax.set_title("ICE Plot : feature_name")
plt.tight_layout()
plt.show()
```

### Interprétation
- **Pente montante** : augmenter la feature augmente la prédiction
- **Pente plate** : la feature n'a pas d'effet dans cette zone
- **Forme non-linéaire** : le modèle capture une relation complexe
- **ICE divergents** : il y a des interactions — la feature agit différemment selon le contexte
- **PDP 2D** : si l'effet d'une feature change selon l'autre → interaction forte

## 3. SHAP Values (SHapley Additive exPlanations)

SHAP est la technique la plus puissante et la plus utilisée dans les solutions gagnantes. Elle décompose chaque prédiction en contributions de chaque feature.

### Formule Fondamentale
```
prediction = base_value + sum(SHAP_values)
```

### TreeExplainer (rapide, exact — pour arbres)

```python
import shap

# Pour modèles basés sur les arbres (XGBoost, LightGBM, CatBoost, RandomForest)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)

# Pour classification binaire : shap_values est une liste [classe_0, classe_1]
# Utiliser shap_values[1] pour la classe positive
# Pour régression : shap_values est un array directement

# Vérification
print(f"Type: {type(shap_values)}")
print(f"Shape: {np.array(shap_values).shape}")
print(f"Base value: {explainer.expected_value}")
```

### Nouvelle API SHAP (v0.40+)

```python
# Nouvelle API unifiée
explainer = shap.TreeExplainer(model)
shap_explanation = explainer(X_val)  # Retourne un objet Explanation

# shap_explanation.values = array SHAP
# shap_explanation.base_values = valeur de base
# shap_explanation.data = données d'entrée
```

### Visualisations SHAP

#### Force Plot (explication d'une prédiction individuelle)

```python
shap.initjs()

# Explication d'une seule prédiction
idx = 0  # Index de l'observation à expliquer
shap.force_plot(
    explainer.expected_value[1],  # [1] pour classification binaire classe positive
    shap_values[1][idx],
    X_val.iloc[idx],
    feature_names=X_val.columns.tolist()
)

# Rouge = pousse la prédiction VERS LE HAUT
# Bleu = pousse la prédiction VERS LE BAS
# Largeur = magnitude de la contribution
```

#### Waterfall Plot (version moderne du force plot)

```python
# Plus lisible que le force plot
shap.waterfall_plot(shap_explanation[idx])
```

#### Summary Plot (vue globale — LE PLUS IMPORTANT)

```python
# Beeswarm plot : chaque point = une observation
shap.summary_plot(shap_values[1], X_val)

# Interprétation :
# - Axe Y : features triées par importance
# - Axe X : impact SHAP (négatif = réduit la prédiction, positif = augmente)
# - Couleur : valeur de la feature (rouge = haute, bleu = basse)
# - Si rouge à droite et bleu à gauche : relation positive
# - Si rouge à gauche et bleu à droite : relation négative
# - Points dispersés : la feature a un impact variable

# Version bar (importance moyenne absolue)
shap.summary_plot(shap_values[1], X_val, plot_type="bar")
```

#### Dependence Plot (interaction entre features)

```python
# Effet d'une feature avec détection automatique d'interaction
shap.dependence_plot(
    'feature_name',
    shap_values[1],
    X_val,
    interaction_index='auto'  # ou spécifier une feature
)

# Interprétation :
# - Axe X : valeur de la feature
# - Axe Y : SHAP value de cette feature
# - Couleur : valeur de la feature d'interaction
# - Dispersion verticale = interactions avec d'autres features
# - Si pas de dispersion : effet indépendant

# Spécifier la feature d'interaction
shap.dependence_plot('feature_A', shap_values[1], X_val,
                     interaction_index='feature_B')
```

#### Decision Plot

```python
# Montre le chemin de décision depuis la base value
shap.decision_plot(
    explainer.expected_value[1],
    shap_values[1][:50],  # 50 premières observations
    X_val.iloc[:50],
    feature_names=X_val.columns.tolist()
)
```

### Explainers pour Autres Types de Modèles

```python
# Deep Learning (PyTorch, TensorFlow)
explainer = shap.DeepExplainer(model, X_train[:100])
shap_values = explainer.shap_values(X_val[:50])

# N'importe quel modèle (model-agnostic — plus lent)
explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 100))
shap_values = explainer.shap_values(X_val[:50])

# Modèles linéaires
explainer = shap.LinearExplainer(model, X_train)
shap_values = explainer.shap_values(X_val)
```

### SHAP pour Feature Engineering (usage Gold Medal)

```python
# Identifier les features à forte interaction
shap_interaction = explainer.shap_interaction_values(X_val)
# shape: (n_samples, n_features, n_features)

# Moyenne des interactions absolues
mean_interactions = np.abs(shap_interaction).mean(axis=0)
interaction_df = pd.DataFrame(mean_interactions, index=X_val.columns, columns=X_val.columns)

# Top interactions
import seaborn as sns
fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(interaction_df, dtype=bool))
sns.heatmap(interaction_df, mask=mask, cmap='YlOrRd', ax=ax, annot=interaction_df.shape[0] <= 15)
ax.set_title('SHAP Feature Interactions')
plt.tight_layout()
plt.show()

# Extraire les top paires d'interaction
interactions_flat = []
for i in range(len(X_val.columns)):
    for j in range(i+1, len(X_val.columns)):
        interactions_flat.append({
            'feature_1': X_val.columns[i],
            'feature_2': X_val.columns[j],
            'interaction_strength': mean_interactions[i, j]
        })

top_interactions = pd.DataFrame(interactions_flat).sort_values(
    'interaction_strength', ascending=False
).head(20)
print("Top 20 interactions de features :")
print(top_interactions.to_string(index=False))

# → Créer des features d'interaction pour les top paires !
# Exemple : df['feat_A_x_feat_B'] = df['feat_A'] * df['feat_B']
```

### SHAP pour Détection de Data Leakage

```python
# Si une feature a une importance SHAP démesurée → suspect de leakage
shap_importance = np.abs(shap_values[1]).mean(axis=0)
shap_imp_df = pd.DataFrame({
    'feature': X_val.columns,
    'mean_abs_shap': shap_importance
}).sort_values('mean_abs_shap', ascending=False)

# Alerter si une feature domine excessivement
top_imp = shap_imp_df.iloc[0]['mean_abs_shap']
second_imp = shap_imp_df.iloc[1]['mean_abs_shap']
if top_imp > 5 * second_imp:
    print(f"⚠ ALERTE : '{shap_imp_df.iloc[0]['feature']}' a une importance 5x supérieure")
    print(f"  → Vérifier s'il y a du data leakage !")
```

## 4. LIME (Local Interpretable Model-agnostic Explanations)

LIME explique des prédictions individuelles en approximant le modèle localement avec un modèle linéaire.

```python
import lime
import lime.lime_tabular

# Créer l'explainer
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=['négatif', 'positif'],
    mode='classification',  # ou 'regression'
    random_state=42
)

# Expliquer une prédiction
idx = 0
exp = lime_explainer.explain_instance(
    X_val.iloc[idx].values,
    model.predict_proba,
    num_features=15,
    num_samples=5000
)

# Afficher
exp.show_in_notebook(show_table=True)

# Récupérer les contributions en DataFrame
lime_df = pd.DataFrame(exp.as_list(), columns=['feature_rule', 'contribution'])
print(lime_df)

# Visualiser
fig = exp.as_pyplot_figure()
plt.tight_layout()
plt.show()
```

### LIME vs SHAP
| Aspect | SHAP | LIME |
|--------|------|------|
| Fondement | Théorie des jeux (Shapley) | Approximation linéaire locale |
| Consistance | Mathématiquement garanti | Approximation (peut varier) |
| Vitesse (arbres) | Très rapide (TreeExplainer) | Plus lent |
| Vitesse (autres) | Lent (KernelExplainer) | Rapide |
| Global vs Local | Les deux | Local seulement |
| Recommandation | Préférer SHAP quand possible | Utile en complément |

## 5. Workflow Complet d'Explainability pour Compétition

```python
def explain_model(model, X_train, X_val, y_val, top_n=30):
    """Pipeline complet d'explainability pour compétition Kaggle."""

    print("=" * 60)
    print("1. PERMUTATION IMPORTANCE")
    print("=" * 60)
    from sklearn.inspection import permutation_importance
    perm = permutation_importance(model, X_val, y_val, n_repeats=10,
                                  random_state=42, n_jobs=-1)
    perm_df = pd.DataFrame({
        'feature': X_val.columns,
        'perm_importance': perm.importances_mean
    }).sort_values('perm_importance', ascending=False)
    print(perm_df.head(top_n).to_string(index=False))

    print("\n" + "=" * 60)
    print("2. SHAP ANALYSIS")
    print("=" * 60)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)

    # Déterminer si classification ou régression
    if isinstance(shap_values, list):
        sv = shap_values[1]  # Classification : classe positive
    else:
        sv = shap_values     # Régression

    # Summary plot
    shap.summary_plot(sv, X_val, show=False)
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    plt.show()

    # Bar plot
    shap.summary_plot(sv, X_val, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.show()

    # Top features SHAP
    shap_imp = pd.DataFrame({
        'feature': X_val.columns,
        'shap_importance': np.abs(sv).mean(axis=0)
    }).sort_values('shap_importance', ascending=False)

    print("\n" + "=" * 60)
    print("3. FEATURES À INVESTIGUER")
    print("=" * 60)

    # Combinaison des deux méthodes
    combined = perm_df.merge(shap_imp, on='feature')
    combined['rank_perm'] = combined['perm_importance'].rank(ascending=False)
    combined['rank_shap'] = combined['shap_importance'].rank(ascending=False)
    combined['avg_rank'] = (combined['rank_perm'] + combined['rank_shap']) / 2
    combined = combined.sort_values('avg_rank')

    print("Top features (consensus SHAP + Permutation) :")
    print(combined.head(top_n)[['feature', 'perm_importance', 'shap_importance', 'avg_rank']].to_string(index=False))

    # Features inutiles à supprimer
    useless = combined[
        (combined['perm_importance'] < 0.001) & (combined['shap_importance'] < 0.001)
    ]
    if len(useless) > 0:
        print(f"\n⚠ {len(useless)} features potentiellement inutiles :")
        print(useless['feature'].tolist())

    print("\n" + "=" * 60)
    print("4. DEPENDENCE PLOTS (Top interactions)")
    print("=" * 60)
    top_features = combined.head(6)['feature'].tolist()
    for feat in top_features:
        shap.dependence_plot(feat, sv, X_val, show=False)
        plt.title(f"SHAP Dependence : {feat}")
        plt.tight_layout()
        plt.show()

    return combined
```

## Checklist Explainability pour Compétition

1. Calculer permutation importance → identifier les features clés
2. SHAP summary plot → comprendre la direction des effets
3. SHAP dependence plots sur les top features → détecter les interactions
4. Créer des features d'interaction basées sur les résultats SHAP
5. Supprimer les features à importance nulle → réduire le bruit
6. Vérifier qu'aucune feature ne domine de façon suspecte → leakage ?
7. PDP/ICE pour valider les relations apprises → sanity check
8. LIME pour expliquer les prédictions aberrantes → debugging

Adapte TOUJOURS l'analyse au type de modèle et à la métrique de la compétition.

## Rapport de Sortie (OBLIGATOIRE)

À la fin de l'analyse d'explainability, TOUJOURS sauvegarder :
1. Rapport dans : `reports/explain/YYYY-MM-DD_explainability.md` (SHAP summary, top features, insights)
2. Confirmer à l'utilisateur le chemin du rapport sauvegardé
