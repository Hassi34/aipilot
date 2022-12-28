import tensorflow as tf
import os

def callbacks (callbacked_model_name ,model_ckpt_path, tensorboard_logs_dir, es_patience=5, lr_patience=3):
    #Tensorboard Callback 
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir = tensorboard_logs_dir)
    #Early stopping callback
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=es_patience, monitor='val_loss',
                                                         restore_best_weights=True)
    #Model Checkpointing callback (Helpful in backup, would save the last checkpoint in crashing)
    CKPT_path = os.path.join( model_ckpt_path ,callbacked_model_name)
    checkpointing_cb = tf.keras.callbacks.ModelCheckpoint(CKPT_path , save_best_only=True)
    reduce_on_plateau_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=lr_patience,
                                                                verbose=1, mode='min', min_lr=0.00000000001)
    return early_stopping_cb, checkpointing_cb, tensorboard_cb, reduce_on_plateau_cb