def main():
  with tf.io.gfile.GFile(config_file_path, 'r') as f:
    config = json.load(f)
    network_config = config['network_config']

  # 1. Shuffle training data set
  # 2. Pick random crop from dataset 
  # 3. Augment Data
  # 4. Repeat for generating validation training set
  
  model = AlphaFoldNetwork(network_config)
  model.compile(
    optimizer=keras.optimizers.SGD(),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)), # from_logits=True means that there is no softmax in the model
    # List of metrics to monitor
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
  )
  print("Fit model on training data")

  callbacks = [
      keras.callbacks.ModelCheckpoint(
          # Path where to save the model
          # The two parameters below mean that we will overwrite
          # the current checkpoint if and only if
          # the `val_loss` score has improved.
          # The saved model name will include the current epoch.
          filepath="mymodel_{epoch}",
          save_best_only=True,  # Only save a model if `val_loss` has improved.
          monitor="val_loss",
          verbose=1,
      )
  ]

  history = model.fit(
      x_train,
      y_train,
      batch_size=4,
      epochs=2,
      # We pass some validation for
      # monitoring validation loss and metrics
      # at the end of each epoch
      validation_data=(x_val, y_val),
      callbacks=callbacks
  )

if __name__ == '__main__':
  main()