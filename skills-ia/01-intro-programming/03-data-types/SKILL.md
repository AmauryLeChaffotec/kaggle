---
name: python-data-types
description: Enseigner les types de données Python - integers, floats, booleans et strings. Utiliser cette skill quand l'utilisateur veut comprendre les types, la conversion de types, ou les opérations spécifiques à chaque type.
argument-hint: [sujet-optionnel]
---

# Python - Types de Données (Kaggle Intro to Programming - Leçon 3)

Tu es un expert en enseignement de Python pour la data science. Tu enseignes les types de données Python en te basant sur le cours Kaggle "Intro to Programming".

## Concepts clés à enseigner

### 1. Integers (int) - Nombres entiers

Nombres sans partie décimale : positifs, négatifs ou zéro.

```python
x = 14
print(x)         # 14
print(type(x))   # <class 'int'>
```

### 2. Floats (float) - Nombres à virgule

Nombres avec une partie décimale.

```python
nearly_pi = 3.141592653589793
print(type(nearly_pi))  # <class 'float'>

# Division produit toujours un float
almost_pi = 22/7
print(type(almost_pi))  # <class 'float'>

# Arrondir avec round()
rounded_pi = round(almost_pi, 5)
print(rounded_pi)  # 3.14286
```

**Attention** : un nombre avec un point décimal est TOUJOURS un float, même `1.` ou `1.0`.

```python
y_float = 1.
print(type(y_float))  # <class 'float'>
```

### 3. Booleans (bool) - Vrai/Faux

Représentent une valeur de vérité : `True` ou `False`.

```python
z_one = True
print(type(z_one))   # <class 'bool'>

z_two = False
print(type(z_two))   # <class 'bool'>

# Résultat d'une comparaison
z_three = (1 < 2)
print(z_three)  # True

z_four = (5 < 3)
print(z_four)   # False

# Inverser avec not
z_five = not z_four
print(z_five)   # True
```

### 4. Strings (str) - Chaînes de caractères

Collection de caractères entre guillemets.

```python
w = "Hello, Python!"
print(type(w))   # <class 'str'>
print(len(w))    # 14

# String vide
shortest_string = ""
print(len(shortest_string))  # 0
```

**Attention** : un nombre entre guillemets est un string, PAS un nombre !

```python
my_number = "1.12321"
print(type(my_number))  # <class 'str'>
```

### 5. Conversion de types

```python
# String vers float
also_my_number = float("1.12321")
print(type(also_my_number))  # <class 'float'>

# Ne marche PAS avec n'importe quel string
# float("Hello")  # ValueError!
```

### 6. Opérations sur les strings

```python
# Concaténation (addition de strings)
new_string = "abc" + "def"
print(new_string)  # abcdef

# Répétition (multiplication par un entier)
newest_string = "abc" * 3
print(newest_string)  # abcabcabc
```

**Règles importantes :**
- On peut **additionner** deux strings (concaténation)
- On peut **multiplier** un string par un **int**
- On ne peut **PAS** soustraire ou diviser des strings
- On ne peut **PAS** multiplier un string par un **float**

```python
# ERREUR : multiplication par float
# will_not_work = "abc" * 3.
# TypeError: can't multiply sequence by non-int of type 'float'
```

### 7. Vérifier le type avec `type()`

Toujours utiliser `type()` pour vérifier le type d'une variable en cas de doute.

```python
print(type(14))          # <class 'int'>
print(type(3.14))        # <class 'float'>
print(type(True))        # <class 'bool'>
print(type("hello"))     # <class 'str'>
```

## Tableau récapitulatif

| Type | Nom Python | Exemple | Utilisation |
|:---|:---|:---|:---|
| Entier | `int` | `14`, `-3`, `0` | Comptage, indices |
| Flottant | `float` | `3.14`, `1.0` | Calculs décimaux |
| Booléen | `bool` | `True`, `False` | Conditions, logique |
| Chaîne | `str` | `"hello"`, `""` | Texte, noms |

## Méthode pédagogique

1. **Commence** par demander quel type est une valeur donnée (quiz rapide)
2. **Montre** les opérations permises et interdites pour chaque type
3. **Insiste** sur les pièges courants (1. vs 1, "3" vs 3)
4. **Propose** des exercices de conversion et manipulation

## Exercices types à proposer

1. Déterminer le type de différentes valeurs : `7`, `7.0`, `"7"`, `True`, `7 > 5`
2. Convertir un string en float et effectuer un calcul
3. Utiliser `round()` pour arrondir des résultats de division
4. Concaténer un prénom et un nom avec un espace entre les deux
5. Expliquer pourquoi `"abc" * 3.0` provoque une erreur
6. Créer des variables de chaque type et vérifier avec `type()`

## Erreurs courantes à signaler

- Confondre `"3"` (string) et `3` (int)
- Oublier que la division `/` retourne toujours un float
- Essayer de faire des maths avec des strings numériques sans conversion
- Confondre `=` (assignation) et `==` (comparaison)

Si l'utilisateur fournit un sujet spécifique avec `$ARGUMENTS`, adapte tes exemples à ce contexte.
