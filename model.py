from tensorflow.keras import layers as L, models as M
from .config import PATCH_SIZE


def conv_block(x, f, residual=True, dropout_rate=0.0):
    skip = x

    x = L.Conv3D(f, 3, padding='same')(x)
    x = L.LayerNormalization()(x)
    x = L.LeakyReLU(0.1)(x)

    x = L.Conv3D(f, 3, padding='same')(x)
    x = L.LayerNormalization()(x)

    if residual:
        if skip.shape[-1] != f:
            skip = L.Conv3D(f, 1, padding='same')(skip)
        x = L.Add()([x, skip])

    x = L.LeakyReLU(0.1)(x)

    if dropout_rate > 0.0:
        x = L.SpatialDropout3D(dropout_rate)(x)

    return x


def build_unet(input_shape=(96, 96, 96, 1), base=32):
    inputs = L.Input(input_shape)
    x = inputs
    skips = []
    f = base

    # Encoder
    for level in range(4):
        x = conv_block(x, f, residual=True, dropout_rate=0.0)
        skips.append(x)
        x = L.MaxPooling3D(2, padding='same')(x)
        f *= 2

    # Bottleneck
    x = conv_block(x, f, residual=True, dropout_rate=0.1)

    # Decoder
    for skip in reversed(skips):
        f //= 2
        x = L.UpSampling3D(2)(x)
        x = L.Concatenate()([x, skip])
        x = conv_block(x, f, residual=True, dropout_rate=0.0)

    outputs = L.Conv3D(1, 1, activation='sigmoid')(x)
    return M.Model(inputs, outputs, name="3D_UNet_Final_NoCrop")
