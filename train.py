import os
import json
import random
import numpy as np
import tensorflow as tf

from .config import (
    TRAIN_IMG_DIR,
    TRAIN_MSK_DIR,
    EPOCHS,
    PATCHES_PER_VOLUME,
    BATCH_SIZE,
    LOG_DIR,
    CKPT_DIR,
)
from .data import match_pairs, make_dataset, make_eval_dataset
from .model import build_unet
from .losses import combo_loss, dice_coef, WarmupCosineDecay


def split_train_val(train_pairs, val_size=5, seed=42):
    rnd = random.Random(seed)
    pairs = list(train_pairs)
    rnd.shuffle(pairs)
    val_pairs = pairs[:val_size]
    trn_pairs = pairs[val_size:]
    return trn_pairs, val_pairs


def train_model(trn_pairs, val_pairs, augment=False, name="baseline"):
    model = build_unet()
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=2e-4,weight_decay=1e-4,clipnorm=0.5),
        loss=combo_loss,
        metrics=[dice_coef]     
    )

    ckpt_path = os.path.join(CKPT_DIR, f"best_{name}.weights.h5")
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        ckpt_path,
        save_best_only=True,
        save_weights_only=True,
        monitor="val_dice_coef",
        mode="max",
    )
    es = tf.keras.callbacks.EarlyStopping(
        patience=10, restore_best_weights=True, monitor="val_dice_coef", mode="max")
    tb = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(LOG_DIR, name))
    csv = tf.keras.callbacks.CSVLogger(os.path.join(LOG_DIR, f"{name}_log.csv"))

    lr_scheduler = WarmupCosineDecay(
        initial_lr=2e-4, warmup_epochs=3, total_epochs=EPOCHS, min_lr=1e-6
    )

    total_patches = len(trn_pairs) * PATCHES_PER_VOLUME
    steps_per_epoch = max(1, int(np.ceil(total_patches / BATCH_SIZE)))
    val_steps = len(val_pairs)

    train_ds = make_dataset(trn_pairs, augment=augment, repeat=True)
    val_ds = make_eval_dataset(val_pairs)

    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        epochs=EPOCHS,
        callbacks=[ckpt, es, tb, csv, lr_scheduler],
        verbose=1,
    )

    hist_path = os.path.join(LOG_DIR, f"{name}_hist.json")
    with open(hist_path, "w") as f:
        json.dump({k: [float(vv) for vv in v] for k, v in hist.history.items()}, f)

    return model


def main(augment=False, name=None):
    train_pairs = match_pairs(TRAIN_IMG_DIR, TRAIN_MSK_DIR)
    trn_pairs, val_pairs = split_train_val(train_pairs, val_size=5)

    if name is None:
        name = "augmented" if augment else "baseline"

    print(f"Train volumes: {len(trn_pairs)} | Val volumes: {len(val_pairs)}")
    print(f"Training model: {name} (augment={augment})")

    train_model(trn_pairs, val_pairs, augment=augment, name=name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--augment", action="store_true", help="Use data augmentation")
    parser.add_argument("--name", type=str, default=None, help="Run name")
    args = parser.parse_args()

    main(augment=args.augment, name=args.name)
