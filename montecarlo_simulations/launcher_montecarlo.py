from collective_learning import CollectiveLearning
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from mpi4py import MPI
from disropt.agents import Agent
from disropt.algorithms import Algorithm
from disropt.utils.graph_constructor import binomial_random_graph, metropolis_hastings

import os

# get MPI info
comm = MPI.COMM_WORLD
nproc = comm.Get_size()
local_rank = comm.Get_rank()

gammas = [0, 1, 10, 100, 1000]
sample_size = 500
reviews = [500]
runs = 1

for gamma in gammas:
    for review in reviews:
        if local_rank==0:
            os.makedirs("{}agents_review{}_samplesize{}_gamma{}".format(nproc, review, sample_size, gamma))
        for run in range(runs):
            if local_rank == 0:
                print("RUN", run)
            ####################################################################
            #####WWWWWWWW########## DATASET LOADING: START #####################
            ####################################################################
            if local_rank == 0:
                print("loading dataset...", end="\r")
            mnist = keras.datasets.fashion_mnist

            (x_train, y_train), (x_test, y_test) = mnist.load_data()  # in ~/.keras/datasets
            x_train = x_train.reshape((60000, 28, 28, 1))
            x_test = x_test.reshape((10000, 28, 28, 1))
            x_train, x_test = x_train / 255.0, x_test / 255.0

            ####################################################################
            ####################### DATASET LOADING: END #######################
            ####################################################################

            ####################################################################
            ############### COMMUNICATION NETWORK CREATION: START ##############
            ####################################################################
            if local_rank == 0:
                print("creating communication network...", end="\r")

            # Generate a common graph (everyone use the same seed)
            Adj = binomial_random_graph(nproc, p=1, seed=1)
            W = metropolis_hastings(Adj)

            # create local agent
            agent = Agent(in_neighbors=np.nonzero(Adj[local_rank, :])[0].tolist(),
                        out_neighbors=np.nonzero(Adj[:, local_rank])[0].tolist(),
                        in_weights=W[local_rank, :].tolist())
            ####################################################################
            ############### COMMUNICATION NETWORK CREATION: END ################
            ####################################################################

            ####################################################################
            ################## LOCAL DATASET CREATION: START ###################
            ####################################################################
            np.random.seed(run)
            if local_rank == 0:
                print("creating local dataset...", end="\r")
            test_sample_number = len(x_test)

            private_samples_training = sample_size
            private_samples_validation = 100
            private_images_training = {}
            private_labels_training = {}
            private_images_validation = {}
            private_labels_validation = {}
            for i in range(nproc):
                indices = np.random.choice(np.arange(len(y_train)), private_samples_training, replace=False)
                private_images_training[i] = x_train[indices]
                private_labels_training[i] = y_train[indices]
                x_train = np.delete(x_train, indices, axis=0)
                y_train = np.delete(y_train, indices, axis=0)

            for i in range(nproc):
                indices = np.random.choice(np.arange(len(y_train)), private_samples_validation, replace=False)
                private_images_validation[i] = x_train[indices]
                private_labels_validation[i] = y_train[indices]
                x_train = np.delete(x_train, indices, axis=0)
                y_train = np.delete(y_train, indices, axis=0)

            shared_set = x_train

            # reset local seed
            np.random.seed()
            ####################################################################
            ################### LOCAL DATASET CREATION: END ####################
            ####################################################################

            ####################################################################
            ################### LOCAL MODEL CREATION: START ####################
            ####################################################################
            tf.keras.backend.clear_session()
            keras.backend.set_floatx('float64')

            CNN = keras.Sequential()
            CNN.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
            CNN.add(keras.layers.MaxPooling2D((2, 2)))
            CNN.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
            CNN.add(keras.layers.MaxPooling2D((2, 2)))
            CNN.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
            CNN.add(keras.layers.Flatten())
            CNN.add(keras.layers.Dense(64, activation='relu'))
            CNN.add(keras.layers.Dense(10, activation='softmax'))

            HL2 = keras.Sequential([
                keras.layers.Flatten(input_shape=(28, 28, 1)),
                keras.layers.Dense(500, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(300, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(10, activation="softmax")
            ])

            HL1 = keras.Sequential([
                keras.layers.Flatten(input_shape=(28, 28, 1)),
                keras.layers.Dense(300, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(10, activation="softmax")
            ])

            SHL = keras.Sequential([
                keras.layers.Flatten(input_shape=(28, 28, 1)),
                keras.layers.Dense(10, activation="softmax")
            ])

            MODELS = [CNN, HL2, HL1, SHL]
            if local_rank == 0:
                print("creating local model...", end="\r")
            rnd = [0, 1, 2, 3]
            optimizer = keras.optimizers.Adam()
            model = MODELS[rnd[local_rank]]

            @tf.function
            def grad(inputs, targets):
                with tf.GradientTape() as tape:
                    predictions = model(inputs)
                    loss_value = keras.losses.sparse_categorical_crossentropy(targets, predictions)
                return loss_value, tape.gradient(loss_value, model.trainable_variables)
            ####################################################################
            ##################### LOCAL MODEL CREATION: END ####################
            ####################################################################

            algo = CollectiveLearning(agent,
                                    model,
                                    grad,
                                    optimizer,
                                    tf.convert_to_tensor(private_images_training[local_rank]),
                                    tf.convert_to_tensor(private_labels_training[local_rank]),
                                    tf.convert_to_tensor(private_images_validation[local_rank]),
                                    tf.convert_to_tensor(private_labels_validation[local_rank]),
                                    tf.convert_to_tensor(shared_set),
                                    tf.convert_to_tensor(x_test),
                                    tf.convert_to_tensor(y_test))

            total_epochs = 3
            accuracies = algo.run(
                epochs=total_epochs,
                self_learning_epochs=5,
                review_epochs=1,
                review_intervals={0: review},
                weights_computation_interval=100,
                test_interval=1000,
                batch_size=10,
                gamma_weight=gamma,
                verbose=True)
            
            if local_rank == 0:
                np.save("{}agents_review{}_samplesize{}_gamma{}/agents.npy".format(nproc, review, sample_size, gamma), nproc)
                np.save("{}agents_review{}_samplesize{}_gamma{}/runs.npy".format(nproc, review, sample_size, gamma), runs)
            np.save("{}agents_review{}_samplesize{}_gamma{}/agent_{}_sequence_run{}.npy".format(nproc, review, sample_size, gamma, local_rank, run), accuracies)
