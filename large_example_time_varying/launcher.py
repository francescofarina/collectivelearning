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
def CNN():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model

def HL2():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28, 1)),
        keras.layers.Dense(500, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(300, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation="softmax")
    ])
    return model

def HL1():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28, 1)),
        keras.layers.Dense(300, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation="softmax")
    ])
    return model

def SHL():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28, 1)),
        keras.layers.Dense(10, activation="softmax")
    ])
    return model


# get MPI info
comm = MPI.COMM_WORLD
nproc = comm.Get_size()
local_rank = comm.Get_rank()

gammas = [100]
sample_size = 100
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
            Adj = binomial_random_graph(nproc, p=0.3, seed=1)
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
            ################### LOCAL MODLE CREATION: START ####################
            ####################################################################
            tf.keras.backend.clear_session()
            keras.backend.set_floatx('float64')

            rnd = np.random.choice([0, 1, 2, 3])
            optimizer = keras.optimizers.Adam()
            if rnd == 0:
                model = CNN()
            elif rnd == 1:
                model = HL2()
            elif rnd == 2:
                model = HL1()
            else:
                model = SHL()

            @tf.function
            def grad(inputs, targets):
                with tf.GradientTape() as tape:
                    predictions = model(inputs)
                    loss_value = keras.losses.sparse_categorical_crossentropy(targets, predictions)
                return loss_value, tape.gradient(loss_value, model.trainable_variables)
            ####################################################################
            ##################### LOCAL MODLE CREATION: END ####################
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
            
            filename = "{}agents_review{}_samplesize{}_gamma{}/agent_{}_sequence_run{}.npy".format(nproc, review, sample_size, gamma, local_rank, run)
            if rnd == 0:
                np.savez(filename, "CNN", accuracies)
            elif rnd == 1:
                np.savez(filename, "HL2", accuracies)
            elif rnd == 2:
                np.savez(filename, "HL1", accuracies)
            else:
                np.savez(filename, "SHL", accuracies)
            
