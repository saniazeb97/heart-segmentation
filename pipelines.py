import tensorflow as tf
import numpy as np
from ..config import PATCHES_PER_VOLUME, BATCH_SIZE, SEED, PATCH_SIZE
from .preprocess import preprocess_pair, preprocess_pair_train
from .patches import extract_patch


def tf_load_volume_train(img_path, msk_path, crop=False):
    def _load(i, m):
        ip = i.decode("utf-8")
        mp = m.decode("utf-8")
        img, msk = preprocess_pair_train(ip, mp, crop=crop)
        return img.astype(np.float32), msk.astype(np.uint8)

    img, msk = tf.numpy_function(_load, [img_path, msk_path], [tf.float32, tf.uint8])
    img.set_shape([None, None, None, 1])
    msk.set_shape([None, None, None, 1])
    return img, msk


def tf_load_volume_eval(img_path, msk_path):
    def _load(i, m):
        ip = i.decode("utf-8")
        mp = m.decode("utf-8")
        img, msk, _ = preprocess_pair(ip, mp, crop=False)
        return img.astype(np.float32), msk.astype(np.uint8)

    img, msk = tf.numpy_function(_load, [img_path, msk_path], [tf.float32, tf.uint8])
    img.set_shape([None, None, None, 1])
    msk.set_shape([None, None, None, 1])
    return img, tf.cast(msk, tf.float32)


def tf_random_patch(img, msk):
    pimg, pmsk = tf.numpy_function(extract_patch, [img, msk], [tf.float32, tf.uint8])
    pimg.set_shape([*PATCH_SIZE, 1])
    pmsk.set_shape([*PATCH_SIZE, 1])
    return pimg, tf.cast(pmsk, tf.float32)


def make_dataset(pairs, augment=False, repeat=False):
    from ..augmentations import safe_tf_augment 

    img_paths = [p[0] for p in pairs]
    msk_paths = [p[1] for p in pairs]

    ds = tf.data.Dataset.from_tensor_slices((img_paths, msk_paths))
    ds = ds.shuffle(len(pairs), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda ip, mp: tf_load_volume_train(ip, mp, crop=False),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds = ds.flat_map(
        lambda i, m: tf.data.Dataset.range(PATCHES_PER_VOLUME).map(lambda _: (i, m))
    )

    ds = ds.map(tf_random_patch, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        ds = ds.map(safe_tf_augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.map(
        lambda i, m: (i, tf.cast(m, tf.float32)), num_parallel_calls=tf.data.AUTOTUNE
    )

    if repeat:
        ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def make_eval_dataset(pairs):
    img_paths = [p[0] for p in pairs]
    msk_paths = [p[1] for p in pairs]

    ds = tf.data.Dataset.from_tensor_slices((img_paths, msk_paths))
    ds = ds.map(
        lambda ip, mp: tf_load_volume_eval(ip, mp),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(1).prefetch(tf.data.AUTOTUNE)
    return ds
