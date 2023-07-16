# Learning Rate Schedulers

This module provides learning rate schedulers that can be used to adjust the learning rate during the training process. Learning rate scheduling is a technique used in deep learning to control the learning rate based on specific criteria such as epochs or predefined schedules.

## LearningRateDecay

The `LearningRateDecay` class is the base class for all learning rate schedulers. It provides a `plot` method that visualizes the learning rate schedule for a given set of epochs.

class LearningRateDecay:
    def plot(self, epochs, title="Learning Rate Schedule"):
        # Code omitted for brevity

## StepDecay

The `StepDecay` class implements a step-based learning rate decay. It reduces the learning rate by a factor at regular intervals defined by the `dropEvery` parameter.

class StepDecay(LearningRateDecay):
    def __init__(self, initAlpha=0.01, factor=0.25, dropEvery=10):
        # Code omitted for brevity

## PolynomialDecay

The `PolynomialDecay` class implements a polynomial learning rate decay. It reduces the learning rate using polynomial decay based on the maximum number of epochs, initial learning rate, and power of the polynomial.

class PolynomialDecay(LearningRateDecay):
    def __init__(self, maxEpochs=100, initAlpha=0.01, power=1.0):
        # Code omitted for brevity

To use these learning rate schedulers, you can import the module into your project and instantiate the desired scheduler class. You can then use the scheduler as a callback in the training process to adjust the learning rate over epochs.

## Example usage

from learning_rate_schedulers import StepDecay

lr_callback = tf.keras.callbacks.LearningRateScheduler(StepDecay(initAlpha=1e-3, dropEvery=20, factor=0.75))
model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
  metrics=['accuracy']
)

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=100,
  callbacks=[lr_callback]
)
