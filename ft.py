import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import register_keras_serializable

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import gc

tf.keras.backend.clear_session()
gc.collect()

from numba import cuda
cuda.select_device(0)
cuda.close()


@register_keras_serializable(package="Custom")
class EncoderOnlyTransformer(tf.keras.Model):
    """
    Transformer encoder migliorato per la predizione multi-step di curve EIS.
    Include Conv1D iniziale, residual connections e LayerNorm finale.
    """

    def __init__(self, 
                 seq_len_in, seq_len_out, 
                 num_features_in, num_features_out,
                 d_model=128, num_heads=4, ff_dim=256,
                 num_encoder_layers=4, dropout=0.2, **kwargs):
        super().__init__(**kwargs)

        # --- Parametri principali ---
        self.seq_len_in = seq_len_in
        self.seq_len_out = seq_len_out
        self.num_features_in = num_features_in
        self.num_features_out = num_features_out
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_encoder_layers = num_encoder_layers
        self.dropout = dropout

        # --- PRE-PROCESSING (nuovo) ---
        self.input_proj = layers.Dense(d_model)
        self.conv_embed = layers.Conv1D(filters=d_model, kernel_size=3, padding='same', activation='relu')
        self.pos_encoding = get_positional_encoding(seq_len_in, d_model)


        # --- ENCODER LAYERS ---
        self.encoder_layers = []
        for _ in range(num_encoder_layers):
            self.encoder_layers.append({
                "norm1": layers.LayerNormalization(epsilon=1e-6),
                "mha": layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout),
                "drop1": layers.Dropout(dropout),
                "norm2": layers.LayerNormalization(epsilon=1e-6),
                "ff1": layers.Dense(ff_dim, activation='relu'),
                "ff2": layers.Dense(d_model),
                "drop2": layers.Dropout(dropout)
            })

        self.final_norm = layers.LayerNormalization(epsilon=1e-6)

        # --- OUTPUT HEAD ---
        #self.global_pool = layers.GlobalAveragePooling1D()
        self.global_pool = layers.Lambda(lambda x: x[:, -1, :])
        self.output_dense = layers.Dense(seq_len_out * num_features_out, use_bias=True)
        self.output_reshape = layers.Reshape((seq_len_out, num_features_out))

    def call(self, inputs, training=False):
        # === INPUT ===
        x = self.input_proj(inputs)
        x = self.conv_embed(x)
        x += self.pos_encoding[:, :tf.shape(x)[1], :]


        # === ENCODER STACK ===
        for layer in self.encoder_layers:
            attn_output = layer["mha"](query=layer["norm1"](x),
                                       value=layer["norm1"](x),
                                       key=layer["norm1"](x),
                                       training=training)
            x = x + layer["drop1"](attn_output, training=training)

            ffn_output = layer["ff2"](layer["ff1"](layer["norm2"](x)))
            x = x + layer["drop2"](ffn_output, training=training)

        x = self.final_norm(x)

        # === OUTPUT ===
        pooled = self.global_pool(x)
        y = self.output_dense(pooled)
        y = self.output_reshape(y)

        shortcut = inputs[:, -1:, :]  # (B, 1, 121)
        # Replica per tutti i timesteps futuri
        shortcut = tf.repeat(shortcut, self.seq_len_out, axis=1)
        
        # Aggiungi solo per le feature di impedenza (0:120)
        y = tf.concat([
            y[:, :, :120] + shortcut[:, :, :120],  # impedenza con residual
            y[:, :, 120:]  # SOH senza residual
        ], axis=-1)
        
        return y

    def get_config(self):
        config = {
            "seq_len_in": self.seq_len_in,
            "seq_len_out": self.seq_len_out,
            "num_features_in": self.num_features_in,
            "num_features_out": self.num_features_out,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "num_encoder_layers": self.num_encoder_layers,
            "dropout": self.dropout
        }
        base_config = super().get_config()
        base_config.update(config)
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
def get_positional_encoding(seq_len, d_model):
    """Restituisce un tensore (1, seq_len, d_model) di embedding sinusoidale."""
    positions = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]  # (seq_len, 1)
    dims = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]       # (1, d_model)
    angle_rates = 1 / tf.pow(10000.0, (2 * (dims // 2)) / tf.cast(d_model, tf.float32))
    angle_rads = positions * angle_rates

    # Applica sin ai canali pari, cos ai dispari
    pos_encoding = tf.concat([tf.sin(angle_rads[:, 0::2]), tf.cos(angle_rads[:, 1::2])], axis=-1)
    pos_encoding = pos_encoding[tf.newaxis, ...]  # (1, seq_len, d_model)
    return pos_encoding


SEQ_LEN_IN = 128
SEQ_LEN_OUT = 10
EPOCHS_BASE = 1000
EPOCHS_FT = 100
BATCH_SIZE = 128

BASE_LR = 1e-4
FT_LR = 1e-5

DATA_PATH = "curves"
SAVE_DIR = "models_10"
os.makedirs(SAVE_DIR, exist_ok=True)

def create_all_sequences(data, seq_len_in, seq_len_out):
    X_enc, y = [], []
    T = len(data)
    win = seq_len_in + seq_len_out
    if T < win:
        return None, None
    for i in range(T - win + 1):
        enc_seq = data[i:i + seq_len_in]
        target = data[i + seq_len_in:i + seq_len_in + seq_len_out]
        X_enc.append(enc_seq)
        y.append(target)
    return np.array(X_enc), np.array(y)

def build_cell_dataset(df, cell_id, feature_cols, seq_len_in, seq_len_out):
    df_cell = df[df["Cell"] == cell_id].copy()
    data = df_cell[feature_cols].values
    scaler = StandardScaler().fit(data)
    data_scaled = scaler.transform(data)
    X_enc, y = create_all_sequences(data_scaled, seq_len_in, seq_len_out)
    return X_enc, y, scaler

def train_model(model, X_train, y_train, X_val, y_val, lr, epochs, save_path):
    optimizer = Adam(lr)
    model.compile(optimizer=optimizer, loss="mae")
    callbacks = [EarlyStopping(patience=15, restore_best_weights=True, monitor="val_loss")]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    model.save(save_path)
    return history

files = [f for f in os.listdir(DATA_PATH) if f.endswith(".csv")]
dataframes = []
for path in files:
    cell = os.path.basename(path).split('.')[0]
    df = pd.read_csv(os.path.join(DATA_PATH, path))
    df.drop(columns=[c for c in df.columns if c.startswith("f_")], inplace=True)
    df.drop(columns=["id", "temperature", "soc", "cell"], errors='ignore', inplace=True)
    df["Cell"] = cell
    dataframes.append(df)
df_all = pd.concat(dataframes, ignore_index=True)
feature_cols = [c for c in df_all.columns if c not in ["Cell"]]

cell_list = sorted(df_all["Cell"].unique())
print(f"Celle trovate: {cell_list}")

base_cell = cell_list[0]
remaining_cells = cell_list[1:]

print(f"\n--- Training base su {base_cell} ---")
X_base, y_base, scaler_base = build_cell_dataset(df_all, base_cell, feature_cols, SEQ_LEN_IN, SEQ_LEN_OUT)


n = len(X_base)
idx = np.arange(n)
np.random.shuffle(idx)
cut = int(0.85 * n)
X_tr, y_tr = X_base[idx[:cut]], y_base[idx[:cut]]
X_va, y_va = X_base[idx[cut:]], y_base[idx[cut:]]

model = EncoderOnlyTransformer(
    seq_len_in=SEQ_LEN_IN,
    seq_len_out=SEQ_LEN_OUT,
    num_features_in=X_tr.shape[-1],
    num_features_out=y_tr.shape[-1],
    d_model=128, num_heads=4, ff_dim=256,
    num_encoder_layers=4, dropout=0.2
)
train_model(model, X_tr, y_tr, X_va, y_va, BASE_LR, EPOCHS_BASE, os.path.join(SAVE_DIR, f"{base_cell}_base.keras"))
dump({base_cell: scaler_base}, os.path.join(SAVE_DIR, f"{base_cell}_scaler.joblib"))

for cell in remaining_cells:
    print(f"\n--- Fine-tuning su {cell} ---")
    X_ft, y_ft, scaler_ft = build_cell_dataset(df_all, cell, feature_cols, SEQ_LEN_IN, SEQ_LEN_OUT)
    
    prev_model_path = os.path.join(SAVE_DIR, f"{base_cell}_base.keras") if cell == remaining_cells[0] \
                      else os.path.join(SAVE_DIR, f"{prev_cell}_ft.keras")
    model = tf.keras.models.load_model(prev_model_path, custom_objects={"EncoderOnlyTransformer": EncoderOnlyTransformer})

    for layer in model.layers[:-3]:
        layer.trainable = False
    
    model.compile(optimizer=Adam(FT_LR), loss="mae")
    
    cb = [EarlyStopping(patience=10, restore_best_weights=True)]
    model.fit(X_ft, y_ft, epochs=EPOCHS_FT, batch_size=BATCH_SIZE, callbacks=cb, verbose=1)
    
    save_path = os.path.join(SAVE_DIR, f"{cell}_ft.keras")
    model.save(save_path)
    dump({cell: scaler_ft}, os.path.join(SAVE_DIR, f"{cell}_scaler.joblib"))
    
    prev_cell = cell

print("\n✅ Training cella-per-cella completato!")
