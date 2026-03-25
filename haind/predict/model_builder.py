"""
LSTM Model Builder
Build and compile LSTM model for earthquake prediction

Author: haind
Date: 2025-03-25
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers
from config import (
    MODEL_DIR, LOG_DIR, INPUT_FEATURES,
    SEQUENCE_LENGTH, BATCH_SIZE, EPOCHS
)


class EarthquakeLSTM:
    """LSTM Model for Earthquake Time & Magnitude Prediction"""

    def __init__(self, n_features, sequence_length=SEQUENCE_LENGTH):
        """
        Initialize LSTM model

        Args:
            n_features: Number of input features (26)
            sequence_length: Number of timesteps (5)
        """
        self.n_features = n_features
        self.sequence_length = sequence_length
        self.model = None
        self.history = None

    def build_model(self, lstm_units=[128, 64, 32], dropout_rate=0.3):
        """
        Build LSTM model with multiple outputs

        Args:
            lstm_units: List of LSTM units for each layer
            dropout_rate: Dropout rate for regularization
        """
        # Input layer
        inputs = layers.Input(shape=(self.sequence_length, self.n_features),
                              name='input_layer')

        # LSTM layers
        x = inputs
        for i, units in enumerate(lstm_units):
            return_sequences = (i < len(lstm_units) - 1)
            x = layers.LSTM(
                units,
                return_sequences=return_sequences,
                name=f'lstm_{i+1}'
            )(x)
            x = layers.Dropout(dropout_rate)(x)
            x = layers.BatchNormalization()(x)

        # Dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)

        # Output 1: Time to next (regression) - positive values only
        output_time = layers.Dense(1, activation='relu', name='output_time')(x)

        # Output 2: Next magnitude (regression) - clip to reasonable range
        output_mag = layers.Dense(1, activation='linear', name='output_mag')(x)
        output_mag = layers.Lambda(lambda x: tf.clip_by_value(x, 0.0, 10.0),
                                  name='mag_clipped')(output_mag)

        # Output 3: Binary classification (M5+ or not)
        output_binary = layers.Dense(1, activation='sigmoid', name='output_binary')(x)

        # Create model
        self.model = models.Model(
            inputs=inputs,
            outputs=[output_time, output_mag, output_binary],
            name='earthquake_lstm'
        )

        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss={
                'output_time': 'mse',
                'output_mag': 'mse',
                'output_binary': 'binary_crossentropy'
            },
            loss_weights={
                'output_time': 1.0,
                'output_mag': 1.0,
                'output_binary': 1.0
            },
            metrics={
                'output_time': ['mae', 'mse'],
                'output_mag': ['mae', 'mse'],
                'output_binary': ['accuracy', 'auc']
            }
        )

        return self.model

    def get_callbacks(self, patience=10):
        """Create training callbacks"""
        return [
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce learning rate on plateau
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            # Model checkpoint
            callbacks.ModelCheckpoint(
                filepath=str(MODEL_DIR / 'best_model.keras'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            # TensorBoard logging
            callbacks.TensorBoard(
                log_dir=str(LOG_DIR),
                histogram_freq=1
            )
        ]

    def summary(self):
        """Print model summary"""
        if self.model is None:
            self.build_model()
        return self.model.summary()

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=EPOCHS, batch_size=BATCH_SIZE):
        """
        Train the model

        Args:
            X_train: Training input
            y_train: Training targets (tuple of 3 arrays)
            X_val: Validation input
            y_val: Validation targets (tuple of 3 arrays)
            epochs: Number of epochs
            batch_size: Batch size

        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()

        # Prepare y as tuple/dict
        y_train_dict = {
            'output_time': y_train[:, 0],
            'output_mag': y_train[:, 1],
            'output_binary': y_train[:, 2]
        }

        validation_data = None
        if X_val is not None and y_val is not None:
            y_val_dict = {
                'output_time': y_val[:, 0],
                'output_mag': y_val[:, 1],
                'output_binary': y_val[:, 2]
            }
            validation_data = (X_val, y_val_dict)

        print(f"\nTraining model...")
        print(f"  Samples: {len(X_train):,}")
        print(f"  Batch size: {batch_size}")
        print(f"  Epochs: {epochs}")

        self.history = self.model.fit(
            X_train,
            y_train_dict,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.get_callbacks(),
            verbose=1
        )

        return self.history

    def predict(self, X):
        """
        Make predictions

        Args:
            X: Input data

        Returns:
            Dictionary with predictions
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load_model() first.")

        predictions = self.model.predict(X)

        return {
            'time_to_next': predictions[0].flatten()[0],
            'next_mag': predictions[1].flatten()[0],
            'next_mag_binary': predictions[2].flatten()[0]
        }

    def save_model(self, filepath=None):
        """Save model to file"""
        if filepath is None:
            filepath = MODEL_DIR / 'earthquake_model.keras'

        self.model.save(filepath)
        print(f"Model saved to: {filepath}")

    def load_model(self, filepath=None):
        """Load model from file"""
        if filepath is None:
            filepath = MODEL_DIR / 'best_model.keras'

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from: {filepath}")

        return self.model


# Test the module
if __name__ == "__main__":
    print("Testing EarthquakeLSTM module...")

    # Get number of features
    from config import INPUT_FEATURES
    n_features = len(INPUT_FEATURES) - 1  # Exclude 'time'

    print(f"Building model with {n_features} features...")

    model_builder = EarthquakeLSTM(n_features=n_features)
    model_builder.build_model()

    print("\nModel Summary:")
    model_builder.summary()

    print("\n✓ EarthquakeLSTM module test completed")
