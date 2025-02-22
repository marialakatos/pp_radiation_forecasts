import tensorflow as tf
import numpy as np

"""
Custom Keras callbacks for training monitoring

Classes:
- TerminateOnNaN: Stops training if NaN values appear in the loss.
- CustomPrintCallback: Prints loss and validation loss at the end of each epoch.

Usage:
    model.fit(..., callbacks=[TerminateOnNaN(), CustomPrintCallback()])
"""

class TerminateOnNaN(tf.keras.callbacks.Callback):
    """Stops training if NaN values are detected in the loss at the end of a batch."""

    def on_batch_end(self, batch, logs=None):
        if logs.get('loss') is not None and np.isnan(logs.get('loss')):
            print(f"NaN detected in batch {batch}. Stopping training.")
            self.model.stop_training = True

class CustomPrintCallback(tf.keras.callbacks.Callback):
    """Prints the loss and validation loss at the end of each epoch."""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print("Epoch {}: Loss: {:.8f} - Validation Loss: {:.8f}".format(epoch, logs.get('loss'), logs.get('val_loss')))