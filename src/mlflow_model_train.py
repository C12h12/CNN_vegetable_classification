import numpy as np
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from glob import glob
import os
import argparse
from get_data import get_data, read_params
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16   #type of cnn model
import tensorflow as tf
import mlflow
from urllib.parse import urlparse
import mlflow.keras

def train_model_mlflow(config_file):
    config = get_data(config_file)
    train = config['model']['trainable']
    if train == True:
        img_size = config['model']['image_size']
        num_class=config['model']['num_class']
        loss=config['model']['loss']
        optimizer=config['model']['optimizer']
        metrics=config['model']['metrics']
        epochs=config['model']['epochs']
        rescale=config['img_augment']['rescale']
        shear_range=config['img_augment']['shear_range']
        zoom_range=config['img_augment']['zoom_range']
        horizontal_flip=config['img_augment']['horizontal_flip']
        vertifal_flip=config['img_augment']['vertical_flip']
        batch=config['img_augment']['batch_size']
        class_mode=config['img_augment']['class_mode']
        artifact_dir=config['mlflow_config']['artifact_dir']
        experiment_name=config['mlflow_config']['experiment_name']
        run_name=config['mlflow_config']['run_name']
        registered_model_name=config['mlflow_config']['registered_model_name']
        remote_server_uri=config['mlflow_config']['remote_server_uri']
        run_name=config['mlflow_config']['run_name']
        train_set=config['processed_data']['train']
        test_set=config['processed_data']['test']
        model_path=config['model']['model_path']

        resnet = VGG16(input_shape=img_size + [3], weights = 'imagenet', include_top = False)
        for p in resnet.layers:
            p.trainable = False

        op = Flatten()(resnet.output)
        prediction = Dense(num_class, activation='softmax')(op)
        mod = Model(inputs = resnet.input, outputs = prediction)

        img_size = tuple(img_size)
        mod.compile(loss = loss, optimizer = optimizer, metrics = metrics)

         #complete from here
        train_gen = ImageDataGenerator(rescale = rescale, 
                                       shear_range = shear_range, 
                                       zoom_range = zoom_range, 
                                       horizontal_flip = horizontal_flip, 
                                       vertical_flip = vertifal_flip,
                                       rotation_range = 90)
        
        test_gen = ImageDataGenerator(rescale = rescale)

        train_set = train_gen.flow_from_directory(train_set,
                                                  target_size = img_size,
                                                  batch_size = batch,
                                                  class_mode = class_mode)
        test_set = test_gen.flow_from_directory(test_set, 
                                                target_size=img_size,
                                                batch_size = batch,
                                                class_mode = class_mode)
        #mlflow code
        mlflow_config = config['mlflow_config']
        remote_server_uri = mlflow_config["remote_server_uri"]
        mlflow.set_tracking_uri(remote_server_uri)
        mlflow.set_experiment(mlflow_config["experiment_name"])
        with mlflow.start_run(run_name=run_name):
            history = mod.fit(train_set,
                          epochs = epochs,
                          validation_data = test_set,
                          steps_per_epoch = len(train_set),
                          validation_steps = len(test_set))
            
            mod.save(model_path)
            print("Model Saved Successfully....!")

            train_loss = history.history['loss'][-1]
            val_loss = history.history['val_loss'][-1]
            val_acc = history.history['val_accuracy'][-1]

            mlflow.log_param("epochs", epochs)   #imp info has been stored in mlflow
            mlflow.log_param("loss", loss)
            mlflow.log_param("val_loss", val_loss)
            mlflow.log_param("val_accuracy", val_acc)
            mlflow.log_param("metrics", val_acc)

            tracking_url_type_Store = urlparse(mlflow.get_artifact_uri()).scheme
            if tracking_url_type_Store != "file":
                mlflow.keras.log_model(mod, "model", registered_model_name=mlflow_config["registered_model_name"])
            else:
                mlflow.keras.log_model(mod, "model")

    else:
        print("Model is not trainable")

if __name__=='__main__':
    args=argparse.ArgumentParser()
    args.add_argument('--config',default='../params.yaml')
    passed_args=args.parse_args()
    train_model_mlflow(config_file=passed_args.config)

    #remember to run the following command to run the script in dlmining folder
    #mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000