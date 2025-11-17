import numpy as np
import tensorflow as tf


def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    inter = tf.reduce_sum(y_true_f * y_pred_f, axis=-1)
    return tf.reduce_mean(
        (2.0 * inter + smooth)
        / (tf.reduce_sum(y_true_f, -1) + tf.reduce_sum(y_pred_f, -1) + smooth)
    )


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def combo_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    return 0.5 * dice_loss(y_true, y_pred) + 0.5 * tf.keras.losses.binary_crossentropy(
        y_true, y_pred
    )


class WarmupCosineDecay(tf.keras.callbacks.Callback):
    def __init__(self, initial_lr, warmup_epochs, total_epochs, min_lr=1e-6):
        super().__init__()
        self.initial_lr = float(initial_lr)
        self.warmup_epochs = int(warmup_epochs)
        self.total_epochs = int(total_epochs)
        self.min_lr = float(min_lr)

    def _lr_at_epoch(self, epoch):
        if epoch < self.warmup_epochs:
            return self.initial_lr * float(epoch + 1) / float(self.warmup_epochs)
        progress = (epoch - self.warmup_epochs) / max(
            1, self.total_epochs - self.warmup_epochs
        )
        return self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (
            1.0 + np.cos(np.pi * progress)
        )

    def on_epoch_begin(self, epoch, logs=None):
        lr = self._lr_at_epoch(epoch)
        if hasattr(self.model.optimizer, "learning_rate"):
            try:
                self.model.optimizer.learning_rate.assign(lr)
            except Exception:
                tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        print(f"\nEpoch {epoch+1}/{self.total_epochs} — LR: {lr:.8f}")
