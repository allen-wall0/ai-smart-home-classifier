import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Reshape, Conv1D, MaxPooling1D, Flatten, MultiHeadAttention, Add, LayerNormalization
from tensorflow.keras import Model

def build_mlp_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        Input(shape=input_shape),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_cnn_model(input_shape, num_classes):
    kernel_size = min(2, input_shape[0])
    model = tf.keras.Sequential([
        Input(shape=input_shape),
        Reshape((input_shape[0], 1)),
        Conv1D(64, kernel_size=kernel_size, activation='relu'),
        MaxPooling1D(pool_size=1),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_transformer_model(input_shape, num_classes):
    input_layer = Input(shape=input_shape)
    x = Reshape((input_shape[0], 1))(input_layer)
    x = Dense(64)(x)
    attention_output = MultiHeadAttention(num_heads=2, key_dim=64)(x, x)
    x = Add()([x, attention_output])
    x = LayerNormalization()(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model