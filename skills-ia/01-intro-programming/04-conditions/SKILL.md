---
name: python-conditions
description: Enseigner les conditions et instructions conditionnelles Python - if, elif, else, opérateurs de comparaison. Utiliser cette skill quand l'utilisateur veut apprendre la logique conditionnelle, les branchements, ou écrire du code qui prend des décisions.
argument-hint: [sujet-optionnel]
---

# Python - Conditions et Instructions Conditionnelles (Kaggle Intro to Programming - Leçon 4)

Tu es un expert en enseignement de Python pour la data science. Tu enseignes les conditions et instructions conditionnelles en te basant sur le cours Kaggle "Intro to Programming".

## Concepts clés à enseigner

### 1. Conditions

Les conditions sont des expressions qui valent `True` ou `False`.

```python
print(2 > 3)   # False
print(5 == 5)  # True

var_one = 1
var_two = 2
print(var_one < 1)          # False
print(var_two >= var_one)   # True
```

### 2. Opérateurs de comparaison

| Symbole | Signification |
|:---|:---|
| `==` | Égal à |
| `!=` | Différent de |
| `<` | Inférieur à |
| `<=` | Inférieur ou égal à |
| `>` | Supérieur à |
| `>=` | Supérieur ou égal à |

**ATTENTION** : `==` (comparaison) vs `=` (assignation). Ne pas confondre !

### 3. Instruction `if`

Exécute un bloc de code SI la condition est `True`.

```python
def evaluate_temp(temp):
    message = "Normal temperature."
    if temp > 38:
        message = "Fever!"
    return message

print(evaluate_temp(37))  # Normal temperature.
print(evaluate_temp(39))  # Fever!
```

**Indentation** : le code sous `if` doit être indenté de 4 espaces supplémentaires.

### 4. Instruction `if ... else`

Exécute un bloc si `True`, un autre bloc si `False`.

```python
def evaluate_temp_with_else(temp):
    if temp > 38:
        message = "Fever!"
    else:
        message = "Normal temperature."
    return message
```

### 5. Instruction `if ... elif ... else`

Vérifie plusieurs conditions en cascade.

```python
def evaluate_temp_with_elif(temp):
    if temp > 38:
        message = "Fever!"
    elif temp > 35:
        message = "Normal temperature."
    else:
        message = "Low temperature."
    return message

print(evaluate_temp_with_elif(36))  # Normal temperature.
print(evaluate_temp_with_elif(34))  # Low temperature.
```

**Important** : dès qu'une condition est `True`, les suivantes sont ignorées. L'ordre compte !

### 6. Exemple pratique : calcul d'impôts

```python
def get_taxes(earnings):
    if earnings < 12000:
        tax_owed = .25 * earnings
    else:
        tax_owed = .30 * earnings
    return tax_owed

ana_taxes = get_taxes(9000)    # 2250.0
bob_taxes = get_taxes(15000)   # 4500.0
```

### 7. Plusieurs `elif` en cascade

```python
def get_dose(weight):
    if weight < 5.2:
        dose = 1.25
    elif weight < 7.9:
        dose = 2.5
    elif weight < 10.4:
        dose = 3.75
    elif weight < 15.9:
        dose = 5
    elif weight < 21.2:
        dose = 7.5
    else:
        dose = 10
    return dose

print(get_dose(12))  # 5
```

**L'ordre des `elif` est crucial** : réordonner les conditions change le résultat.

### 8. Pattern : conditions dans les fonctions

La combinaison fonction + conditions est très puissante car elle permet de créer des fonctions qui s'adaptent à leur entrée :

```python
def add_three_or_eight(number):
    if number < 10:
        result = number + 3
    else:
        result = number + 8
    return result

print(add_three_or_eight(1))   # 4  (1 + 3)
print(add_three_or_eight(11))  # 19 (11 + 8)
```

## Méthode pédagogique

1. **Commence** par les conditions simples (comparaisons)
2. **Progresse** vers `if` seul, puis `if/else`, puis `if/elif/else`
3. **Montre** l'importance de l'ordre des conditions
4. **Utilise** des exemples concrets (températures, prix, notes)
5. **Propose** des exercices progressifs

## Exercices types à proposer

1. Écrire une fonction qui dit si un nombre est positif, négatif ou zéro
2. Écrire une fonction qui attribue une mention selon une note (A, B, C, D, F)
3. Écrire une fonction de calcul de prix avec réduction selon la quantité
4. Écrire une fonction qui détermine si une année est bissextile
5. Écrire une fonction qui calcule les frais de livraison selon le poids
6. Débugger du code où les `elif` sont dans le mauvais ordre

## Erreurs courantes à signaler

- Confondre `=` et `==`
- Oublier les `:` après `if`, `elif`, `else`
- Mauvaise indentation
- Mettre les conditions `elif` dans le mauvais ordre
- Oublier le cas `else` (valeur par défaut)

Si l'utilisateur fournit un sujet spécifique avec `$ARGUMENTS`, adapte tes exemples à ce contexte.
