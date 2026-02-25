---
name: kaggle-tpu
description: Expert TPU & TensorFlow/Keras pour compétitions Kaggle. Utiliser quand l'utilisateur travaille avec TensorFlow, Keras, TPU, TFRecords, tf.data, ou des compétitions nécessitant une stratégie de distribution TPU.
argument-hint: <type de tâche TF/TPU ou compétition>
---

# Expert TPU & TensorFlow — Kaggle

Tu es un expert en TensorFlow/Keras et TPU pour les compétitions Kaggle. Tu maîtrises tf.distribute, tf.data, TFRecords, TTA, les LR schedules Keras, et l'API Fonctionnelle Keras.

## Stack Technique

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import numpy as np
import pandas as pd
import re, os, random
from functools import partial
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
```

## 1. TPU Setup & Distribution Strategy

### Détection et Initialisation TPU

```python
# Détection automatique TPU / GPU / CPU
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU:', tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    DEVICE = "TPU"
except ValueError:
    strategy = tf.distribute.get_strategy()  # CPU ou single GPU
    DEVICE = "GPU" if len(tf.config.experimental.list_physical_devices('GPU')) > 0 else "CPU"

AUTO = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync
print(f"Device: {DEVICE} | Replicas: {REPLICAS}")
```

### Batch Size Scaling

```python
# RÈGLE : batch_size = base × num_replicas
BASE_BATCH_SIZE = 16
BATCH_SIZE = BASE_BATCH_SIZE * strategy.num_replicas_in_sync  # 16 × 8 = 128 sur TPU v3-8

# Steps per epoch
NUM_TRAIN = count_data_items(TRAIN_FILES)
NUM_VALID = count_data_items(VALID_FILES)
STEPS_PER_EPOCH = NUM_TRAIN // BATCH_SIZE
VALID_STEPS = -(-NUM_VALID // BATCH_SIZE)  # ceil division
```

### Construire un modèle dans strategy.scope()

```python
# OBLIGATOIRE : tout ce qui crée des variables TF doit être dans scope()
with strategy.scope():
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['AUC']
    )
```

## 2. TFRecords — Format Standard Kaggle

### Compter les éléments depuis les noms de fichiers

```python
def count_data_items(filenames):
    """Comptage rapide depuis le pattern de nommage Kaggle : xxx-NNN.tfrec"""
    n = [int(re.compile(r"-([0-9]*)\.").search(fn).group(1)) for fn in filenames]
    return np.sum(n)
```

### Parsing de TFRecords

```python
def read_labeled_tfrecord(example):
    """Parse un TFRecord labellé (train/valid)."""
    schema = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'image_id': tf.io.FixedLenFeature([], tf.string),
        'target': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, schema)
    image = decode_image(example['image'])
    label = tf.cast(example['target'], tf.int32)
    return image, label

def read_unlabeled_tfrecord(example, return_image_id=True):
    """Parse un TFRecord non labellé (test)."""
    schema = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'image_id': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, schema)
    image = decode_image(example['image'])
    id_or_zero = example['image_id'] if return_image_id else 0
    return image, id_or_zero
```

### Décodage d'images

```python
def decode_image(image_data, image_size=(512, 512)):
    """Decode JPEG/PNG depuis un TFRecord."""
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [*image_size, 3])
    return image
```

## 3. tf.data Pipeline Optimisé

### Pipeline Complet

```python
def get_dataset(files, augment=False, shuffle=False, repeat=False,
                labeled=True, batch_size=BATCH_SIZE, dim=(512, 512)):
    """Pipeline tf.data optimisé pour TPU."""

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)
    ds = ds.cache()  # Cache en RAM après premier read

    if repeat:
        ds = ds.repeat()

    if shuffle:
        ds = ds.shuffle(2048, seed=SEED)
        opt = tf.data.Options()
        opt.experimental_deterministic = False  # Parallélisme non-déterministe = plus rapide
        ds = ds.with_options(opt)

    # Parse
    if labeled:
        ds = ds.map(read_labeled_tfrecord, num_parallel_calls=AUTO)
    else:
        ds = ds.map(read_unlabeled_tfrecord, num_parallel_calls=AUTO)

    # Augmentation
    if augment:
        ds = ds.map(lambda img, lbl: (augment_image(img, dim), lbl), num_parallel_calls=AUTO)

    ds = ds.batch(batch_size * REPLICAS)
    ds = ds.prefetch(AUTO)  # Précharge le prochain batch pendant le training
    return ds
```

### Pipelines Train / Valid / Test

```python
def get_training_dataset():
    ds = get_dataset(TRAIN_FILES, augment=True, shuffle=True, repeat=True, labeled=True)
    return ds

def get_validation_dataset():
    ds = get_dataset(VALID_FILES, augment=False, shuffle=False, repeat=False, labeled=True)
    return ds

def get_test_dataset():
    ds = get_dataset(TEST_FILES, augment=False, shuffle=False, repeat=False, labeled=False)
    return ds
```

### Ordre des Opérations tf.data (IMPORTANT)

```
TFRecordDataset → .cache() → .repeat() → .shuffle() → .map(parse) → .map(augment) → .batch() → .prefetch()
```

| Opération | Pourquoi |
|-----------|----------|
| `.cache()` | Stocke en RAM/disque après le premier parcours, évite re-lecture |
| `.repeat()` | Boucle infinie pour `steps_per_epoch` |
| `.shuffle(buffer)` | Buffer ≥ 2048, AVANT batch pour mélanger les exemples |
| `.map(fn, num_parallel_calls=AUTO)` | Parse et augmente en parallèle sur CPU |
| `.batch(bs * REPLICAS)` | Global batch size = per-replica × num_replicas |
| `.prefetch(AUTO)` | Précharge le prochain batch pendant le GPU/TPU compute |

## 4. Augmentation TensorFlow

### Augmentations Basiques

```python
def augment_image(image, dim=(512, 512)):
    """Augmentations standard compatible TPU."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_saturation(image, 0.7, 1.3)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.reshape(image, [*dim, 3])
    return image
```

### Coarse Dropout (CutOut)

```python
def cutout(image, dim=(512, 512), probability=0.6, count=5, size=0.1):
    """Coarse Dropout : masque des zones aléatoires de l'image."""
    if tf.random.uniform([]) > probability:
        return image
    for _ in range(count):
        x = tf.cast(tf.random.uniform([], 0, dim[1]), tf.int32)
        y = tf.cast(tf.random.uniform([], 0, dim[0]), tf.int32)
        width = tf.cast(size * min(dim), tf.int32)
        ya, yb = tf.math.maximum(0, y - width//2), tf.math.minimum(dim[0], y + width//2)
        xa, xb = tf.math.maximum(0, x - width//2), tf.math.minimum(dim[1], x + width//2)
        left   = image[ya:yb, 0:xa, :]
        middle = tf.zeros([yb - ya, xb - xa, 3], dtype=image.dtype)
        right  = image[ya:yb, xb:dim[1], :]
        row    = tf.concat([left, middle, right], axis=1)
        image  = tf.concat([image[0:ya, :, :], row, image[yb:dim[0], :, :]], axis=0)
    return tf.reshape(image, [*dim, 3])
```

### Affine Transform (Rotation, Shear, Zoom, Shift)

```python
import math
import tensorflow.keras.backend as K

def get_transform_matrix(rotation, shear, h_zoom, w_zoom, h_shift, w_shift):
    """Matrice de transformation affine 3×3."""
    rotation = math.pi * rotation / 180.
    shear = math.pi * shear / 180.

    def mat3x3(lst):
        return tf.reshape(tf.concat([lst], axis=0), [3, 3])

    one, zero = tf.constant([1.], tf.float32), tf.constant([0.], tf.float32)
    c1, s1 = tf.math.cos(rotation), tf.math.sin(rotation)
    c2, s2 = tf.math.cos(shear), tf.math.sin(shear)

    rot   = mat3x3([c1, s1, zero, -s1, c1, zero, zero, zero, one])
    shr   = mat3x3([one, s2, zero, zero, c2, zero, zero, zero, one])
    zoom  = mat3x3([one/h_zoom, zero, zero, zero, one/w_zoom, zero, zero, zero, one])
    shift = mat3x3([one, zero, h_shift, zero, one, w_shift, zero, zero, one])
    return K.dot(K.dot(rot, shr), K.dot(zoom, shift))

def spatial_transform(image, dim=(512, 512),
                      rot=10.0, shr=5.0, hzoom=8.0, wzoom=8.0, hshift=8.0, wshift=8.0):
    """Applique rotation + shear + zoom + shift aléatoires."""
    rot_val   = rot * tf.random.normal([1])
    shr_val   = shr * tf.random.normal([1])
    h_zoom    = 1.0 + tf.random.normal([1]) / hzoom
    w_zoom    = 1.0 + tf.random.normal([1]) / wzoom
    h_shift   = hshift * tf.random.normal([1])
    w_shift   = wshift * tf.random.normal([1])
    m = get_transform_matrix(rot_val, shr_val, h_zoom, w_zoom, h_shift, w_shift)

    DIM = dim[0]
    x = tf.repeat(tf.range(DIM//2, -DIM//2, -1), DIM)
    y = tf.tile(tf.range(-DIM//2, DIM//2), [DIM])
    z = tf.ones([DIM * DIM], dtype='int32')
    idx = tf.stack([x, y, z])
    idx2 = K.dot(m, tf.cast(idx, tf.float32))
    idx2 = K.cast(idx2, 'int32')
    idx2 = K.clip(idx2, -DIM//2 + 1, DIM//2)
    idx3 = tf.stack([DIM//2 - idx2[0,], DIM//2 - 1 + idx2[1,]])
    image = tf.gather_nd(image, tf.transpose(idx3))
    return tf.reshape(image, [*dim, 3])
```

### Preprocessing Layers (Keras intégré)

```python
from tensorflow.keras.layers.experimental import preprocessing

augmentation_layers = keras.Sequential([
    preprocessing.RandomFlip('horizontal'),
    preprocessing.RandomRotation(0.1),
    preprocessing.RandomContrast(0.3),
    preprocessing.RandomZoom(0.1),
])
```

## 5. Test-Time Augmentation (TTA)

### TTA Pattern Compétition

```python
TTA_STEPS = 11  # Nombre de passes augmentées (impair recommandé)

def predict_with_tta(model, dataset_fn, num_samples, batch_size, tta_steps=TTA_STEPS):
    """Prédiction avec TTA : moyenne de N passes augmentées."""
    # Dataset avec augmentation + repeat pour N passes
    ds = dataset_fn(augment=True, repeat=True, shuffle=False, labeled=False)

    steps = tta_steps * num_samples / batch_size / REPLICAS
    raw_preds = model.predict(ds, steps=steps, verbose=1)[:tta_steps * num_samples]

    # Reshape (num_samples, tta_steps) puis moyenne
    preds = raw_preds.reshape((num_samples, tta_steps), order='F')
    return np.mean(preds, axis=1)

# Usage
# Validation OOF avec TTA
oof_preds = predict_with_tta(model, get_val_dataset, NUM_VALID, BATCH_SIZE)

# Test avec TTA
test_preds = predict_with_tta(model, get_test_dataset, NUM_TEST, BATCH_SIZE)
```

### TTA Multi-fold avec Poids

```python
TTA = 11
FOLDS = 5
WGTS = [1/FOLDS] * FOLDS  # Poids égaux par défaut

preds = np.zeros((NUM_TEST, 1))
oof_pred, oof_tar = [], []

for fold in range(FOLDS):
    model = build_model()
    model.load_weights(f'fold-{fold}.h5')

    # OOF avec TTA
    ds_val = get_dataset(val_files[fold], augment=True, repeat=True,
                         shuffle=False, labeled=False, batch_size=BATCH_SIZE * 2)
    ct_val = count_data_items(val_files[fold])
    steps = TTA * ct_val / BATCH_SIZE / 2 / REPLICAS
    pred = model.predict(ds_val, steps=steps)[:TTA * ct_val]
    oof_pred.append(np.mean(pred.reshape((ct_val, TTA), order='F'), axis=1))

    # Test avec TTA
    ds_test = get_dataset(test_files, augment=True, repeat=True,
                          shuffle=False, labeled=False, batch_size=BATCH_SIZE * 2)
    ct_test = count_data_items(test_files)
    steps = TTA * ct_test / BATCH_SIZE / 2 / REPLICAS
    pred = model.predict(ds_test, steps=steps)[:TTA * ct_test]
    preds[:, 0] += np.mean(pred.reshape((ct_test, TTA), order='F'), axis=1) * WGTS[fold]

# OOF score
oof = np.concatenate(oof_pred)
true = np.concatenate(oof_tar)
print(f"OOF AUC: {roc_auc_score(true, oof):.5f}")
```

## 6. Learning Rate Schedules Keras

### Warmup + Exponential Decay (Pattern Gold Medal)

```python
def get_lr_callback(batch_size=8, replicas=REPLICAS, plot=False):
    """LR schedule : warmup ramp → sustain → exponential decay."""
    lr_start   = 1e-6
    lr_max     = 1.25e-6 * replicas * batch_size  # Scale avec batch size
    lr_min     = 1e-6
    lr_ramp_ep = 5      # Epochs de warmup
    lr_sus_ep  = 0      # Epochs à lr_max constant
    lr_decay   = 0.8    # Facteur de decay

    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            # Warmup linéaire
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep:
            # Sustain
            lr = lr_max
        else:
            # Exponential decay
            lr = (lr_max - lr_min) * lr_decay ** (epoch - lr_ramp_ep - lr_sus_ep) + lr_min
        return lr

    if plot:
        import matplotlib.pyplot as plt
        epochs = np.arange(30)
        plt.figure(figsize=(10, 4))
        plt.plot(epochs, [lrfn(e) for e in epochs], marker='o')
        plt.xlabel('Epoch'); plt.ylabel('Learning Rate')
        plt.title('LR Schedule'); plt.show()

    return tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
```

### Cosine Decay avec Warmup

```python
def cosine_lr_callback(total_epochs, warmup_epochs=5, lr_max=1e-3, lr_min=1e-6):
    """Cosine decay avec warmup linéaire."""
    def lrfn(epoch):
        if epoch < warmup_epochs:
            return lr_min + (lr_max - lr_min) * epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * progress))

    return tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
```

### ExponentialDecay (API intégrée)

```python
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-5,
    decay_steps=10000,
    decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-3)
```

### ReduceLROnPlateau

```python
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-6,
    verbose=1
)
```

## 7. Callbacks Essentiels

### Callbacks Compétition

```python
def get_callbacks(fold, monitor='val_auc', mode='max'):
    """Callbacks standard pour compétition Kaggle."""
    return [
        # Sauvegarder le meilleur modèle
        tf.keras.callbacks.ModelCheckpoint(
            f'fold-{fold}.h5',
            monitor=monitor,
            save_best_only=True,
            save_weights_only=True,
            mode=mode,
            verbose=0
        ),
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=5,
            mode=mode,
            restore_best_weights=True,
            verbose=1
        ),
        # LR schedule
        get_lr_callback(batch_size=BATCH_SIZE),
    ]
```

### Callback Personnalisé

```python
class CompetitionLogger(tf.keras.callbacks.Callback):
    """Log des métriques par epoch pour analyse."""
    def __init__(self):
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        self.history.append({**logs, 'epoch': epoch, 'lr': float(self.model.optimizer.lr)})
        print(f"Epoch {epoch}: loss={logs['loss']:.4f} val_auc={logs.get('val_auc', 0):.5f} "
              f"lr={float(self.model.optimizer.lr):.2e}")
```

## 8. Keras Functional API

### Multi-Input Model

```python
def build_multi_input_model(image_size=(512, 512), num_tabular_features=10, num_classes=5):
    """Modèle combinant image + features tabulaires."""
    # Branche Image
    img_input = keras.Input(shape=(*image_size, 3), name='image')
    base = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet')
    base.trainable = False
    x_img = base(img_input)
    x_img = layers.GlobalAveragePooling2D()(x_img)
    x_img = layers.Dense(128, activation='relu')(x_img)

    # Branche Tabulaire
    tab_input = keras.Input(shape=(num_tabular_features,), name='tabular')
    x_tab = layers.Dense(64, activation='relu')(tab_input)
    x_tab = layers.BatchNormalization()(x_tab)
    x_tab = layers.Dense(32, activation='relu')(x_tab)

    # Fusion
    merged = layers.Concatenate()([x_img, x_tab])
    x = layers.Dense(64, activation='relu')(merged)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=[img_input, tab_input], outputs=output)
    return model
```

### Wide & Deep Network (Tabular)

```python
def build_wide_and_deep(input_dim, units=1024, dropout=0.3, activation='relu'):
    """Wide & Deep : combinaison linéaire + DNN profond."""

    def dense_block(x, units, activation, dropout_rate):
        x = layers.Dense(units)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        x = layers.Dropout(dropout_rate)(x)
        return x

    with strategy.scope():
        # Wide : modèle linéaire
        wide = keras.experimental.LinearModel()

        # Deep : DNN multicouches
        inputs = keras.Input(shape=[input_dim])
        x = dense_block(inputs, units, activation, dropout)
        x = dense_block(x, units, activation, dropout)
        x = dense_block(x, units, activation, dropout)
        x = dense_block(x, units, activation, dropout)
        x = dense_block(x, units, activation, dropout)
        outputs = layers.Dense(1)(x)
        deep = keras.Model(inputs=inputs, outputs=outputs)

        # Combinaison
        model = keras.experimental.WideDeepModel(
            linear_model=wide,
            dnn_model=deep,
            activation='sigmoid'
        )

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['AUC', 'binary_accuracy'],
    )
    return model
```

### Multi-Output Model

```python
def build_multi_output(input_shape, n_classes_a=5, n_classes_b=3):
    """Modèle avec deux têtes de sortie."""
    inputs = keras.Input(shape=input_shape)

    # Backbone partagé
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)

    # Tête A
    out_a = layers.Dense(n_classes_a, activation='softmax', name='output_a')(x)

    # Tête B
    out_b = layers.Dense(n_classes_b, activation='softmax', name='output_b')(x)

    model = keras.Model(inputs=inputs, outputs=[out_a, out_b])
    model.compile(
        optimizer='adam',
        loss={'output_a': 'sparse_categorical_crossentropy',
              'output_b': 'sparse_categorical_crossentropy'},
        loss_weights={'output_a': 1.0, 'output_b': 0.5},
        metrics=['accuracy']
    )
    return model
```

## 9. Transfer Learning TensorFlow

### EfficientNet avec Fine-tuning Progressif

```python
def build_efficientnet(dim=(512, 512), ef='B0', num_classes=5):
    """EfficientNet avec GlobalAveragePooling."""
    EFNS = {
        'B0': tf.keras.applications.EfficientNetB0,
        'B1': tf.keras.applications.EfficientNetB1,
        'B2': tf.keras.applications.EfficientNetB2,
        'B3': tf.keras.applications.EfficientNetB3,
        'B4': tf.keras.applications.EfficientNetB4,
        'B5': tf.keras.applications.EfficientNetB5,
        'B6': tf.keras.applications.EfficientNetB6,
        'B7': tf.keras.applications.EfficientNetB7,
    }

    with strategy.scope():
        base = EFNS[ef](input_shape=(*dim, 3), include_top=False, weights='imagenet')
        base.trainable = True  # Fine-tune tout

        model = keras.Sequential([
            base,
            layers.GlobalAveragePooling2D(),
            layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy']
        )
    return model
```

### ResNet50 Transfer Learning

```python
with strategy.scope():
    preprocess = tf.keras.layers.Lambda(
        tf.keras.applications.resnet50.preprocess_input,
        input_shape=[*IMAGE_SIZE, 3]
    )
    base = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)
    base.trainable = False  # Geler d'abord

    model = tf.keras.Sequential([
        tf.keras.layers.BatchNormalization(renorm=True),
        preprocess,
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
```

## 10. Pipeline Complet Compétition — K-Fold + TTA

```python
# === CONFIG ===
SEED = 42
FOLDS = 5
EPOCHS = [20] * FOLDS
IMG_SIZES = [(512, 512)] * FOLDS
BATCH_SIZES = [16] * FOLDS
EFF_NETS = ['B4'] * FOLDS
TTA = 11
AUGMENT = True

# === SEEDING ===
def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = str(seed)
    tf.random.set_seed(seed)

seed_everything(SEED)

# === K-FOLD TRAINING ===
skf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
oof_pred, oof_tar, oof_folds = [], [], []
preds = np.zeros((count_data_items(TEST_FILES), 1))

for fold, (idx_train, idx_valid) in enumerate(skf.split(np.arange(len(TRAIN_FILES)))):
    print(f"\n{'='*40} FOLD {fold} {'='*40}")

    # Fichiers train/valid pour ce fold
    files_train = [TRAIN_FILES[i] for i in idx_train]
    files_valid = [TRAIN_FILES[i] for i in idx_valid]
    ct_train = count_data_items(files_train)
    ct_valid = count_data_items(files_valid)

    # Build model
    model = build_efficientnet(dim=IMG_SIZES[fold], ef=EFF_NETS[fold])

    # Callbacks
    cbs = [
        tf.keras.callbacks.ModelCheckpoint(
            f'fold-{fold}.h5', monitor='val_auc',
            save_best_only=True, save_weights_only=True, mode='max'),
        get_lr_callback(BATCH_SIZES[fold]),
    ]

    # Train
    history = model.fit(
        get_dataset(files_train, augment=AUGMENT, shuffle=True, repeat=True,
                    dim=IMG_SIZES[fold], batch_size=BATCH_SIZES[fold]),
        epochs=EPOCHS[fold],
        callbacks=cbs,
        steps_per_epoch=ct_train // BATCH_SIZES[fold] // REPLICAS,
        validation_data=get_dataset(files_valid, augment=False, shuffle=False,
                                    repeat=False, dim=IMG_SIZES[fold]),
        verbose=1
    )

    # Load best weights
    model.load_weights(f'fold-{fold}.h5')

    # OOF predictions avec TTA
    ds_val = get_dataset(files_valid, augment=AUGMENT, repeat=True,
                         shuffle=False, labeled=False, batch_size=BATCH_SIZES[fold] * 2,
                         dim=IMG_SIZES[fold])
    steps = TTA * ct_valid / BATCH_SIZES[fold] / 2 / REPLICAS
    pred = model.predict(ds_val, steps=steps)[:TTA * ct_valid]
    oof_pred.append(np.mean(pred.reshape((ct_valid, TTA), order='F'), axis=1))

    # Test predictions avec TTA
    ds_test = get_dataset(TEST_FILES, augment=AUGMENT, repeat=True,
                          shuffle=False, labeled=False, batch_size=BATCH_SIZES[fold] * 2,
                          dim=IMG_SIZES[fold])
    ct_test = count_data_items(TEST_FILES)
    steps = TTA * ct_test / BATCH_SIZES[fold] / 2 / REPLICAS
    pred = model.predict(ds_test, steps=steps)[:TTA * ct_test]
    preds[:, 0] += np.mean(pred.reshape((ct_test, TTA), order='F'), axis=1) / FOLDS

# === OOF SCORE ===
oof = np.concatenate(oof_pred)
true = np.concatenate(oof_tar)
print(f"\n{'='*40}")
print(f"Overall OOF AUC: {roc_auc_score(true, oof):.5f}")

# === SUBMISSION ===
sub = pd.DataFrame({'id': test_ids, 'target': preds[:, 0]})
sub = sub.sort_values('id')
sub.to_csv('submission.csv', index=False)
print(f"Submission shape: {sub.shape}")
sub.head()
```

## 11. Quand Utiliser TensorFlow vs PyTorch

| Critère | TensorFlow/Keras | PyTorch |
|---------|-----------------|---------|
| **TPU disponible** | Recommandé | Support limité (XLA) |
| **Gros dataset image** | tf.data + TFRecords optimal | DataLoader moins performant sur TPU |
| **Prototypage rapide** | Keras Sequential/Functional | Plus flexible |
| **Solutions gagnantes** | ~30% des top solutions | ~70% des top solutions |
| **NLP (Transformers)** | HuggingFace TF | HuggingFace PT (plus mature) |
| **Tabular** | Rarement utilisé | Rarement utilisé (GBDT préféré) |
| **Multi-GPU** | tf.distribute.MirroredStrategy | torch.distributed |

### Règle Compétition

- **TPU obligatoire** (compétitions image avec datasets > 50GB) → TensorFlow
- **TPU optionnel** → PyTorch souvent plus simple
- **Code-only competitions** avec TPU fourni → TensorFlow + tf.data + TFRecords
- **Ensemble final** → combiner modèles TF et PyTorch pour diversité maximale

Utilise TOUJOURS `strategy.scope()` pour construire les modèles TPU et scale le batch size avec `REPLICAS`.

## Rapport de Sortie (OBLIGATOIRE)

À la fin de l'analyse, TOUJOURS sauvegarder :
1. Rapport dans : `reports/tpu/YYYY-MM-DD_<description>.md`
2. Contenu : stratégie recommandée, techniques clés, code snippets, recommandations
3. Confirmer à l'utilisateur le chemin du rapport sauvegardé
