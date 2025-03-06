from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

def create_model(input_shape):
    model = Sequential([
        Conv1D(32, kernel_size=3, strides=1, padding='same', input_shape=input_shape,
               kernel_regularizer=regularizers.l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Conv1D(64, kernel_size=3, strides=1, padding='same',
               kernel_regularizer=regularizers.l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Flatten(),

        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.5),

        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.5),

        Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.5),

        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.0005), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])

    return model
