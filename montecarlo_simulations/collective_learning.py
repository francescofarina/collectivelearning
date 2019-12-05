import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from mpi4py import MPI
from disropt.agents import Agent
from disropt.algorithms import Algorithm
from disropt.utils.graph_constructor import binomial_random_graph, metropolis_hastings


class CollectiveLearning(Algorithm):
        def __init__(self, agent: Agent, model=None, grad=None, optimizer=None, X_train=None, Y_train=None, X_val=None,
                     Y_val=None, X_shared=None, X_test=None, Y_test=None, enable_log: bool = True):
            self.agent = agent
            self.model = model
            self.grad = grad
            self.optimizer = optimizer
            self.stepsize = tf.Variable(0)
            self.enable_log = enable_log

            self.val_accuracy = 0.0

            self.metric = tf.keras.metrics.Accuracy()
            self.test_metric = tf.keras.metrics.Accuracy()

            self.X_train = X_train
            self.Y_train = Y_train
            self.training_samples = len(X_train)
            self.X_val = X_val
            self.Y_val = Y_val
            self.X_shared = X_shared
            self.X_test = X_test
            self.Y_test = Y_test

        def self_learning(self, epochs=5, batch_size=10):
            batches = self.training_samples // batch_size
            for ep in range(epochs):
                for k in range(batches):
                    x = tf.reshape(self.X_train[k*batch_size:(k+1)*batch_size], (batch_size, 28, 28, 1))
                    y = tf.cast(self.Y_train[k*batch_size:(k+1)*batch_size], tf.int32)
                    loss_value, grads = self.grad(x, y)
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables), self.stepsize)

        def eval_accuracy(self, print_log=False):
            prediction = tf.argmax(self.model(self.X_val), axis=1, output_type=tf.int32)
            self.validation_accuracy = self.metric(prediction, self.Y_val)
            if print_log:
                print(self.agent.id, self.validation_accuracy)
            self.metric.reset_states()

        def change_weights(self, gamma_weight=100):
            exp_weight = tf.exp(gamma_weight * self.validation_accuracy)
            data = self.agent.neighbors_exchange(exp_weight)
            sum_score = exp_weight + sum(data.values())

            self.agent.in_weights[self.agent.id] = float(exp_weight/sum_score)
            for neigh in self.agent.in_neighbors:
                self.agent.in_weights[neigh] = float(data[neigh]/sum_score)

        def collective_prediction(self, samples, batch_size):
            prediction = tf.one_hot(tf.argmax(
                self.model(tf.reshape(samples, (batch_size, 28, 28, 1))),
                axis=1, output_type=tf.int32),
                10)
            data = self.agent.neighbors_exchange(prediction)
            data[self.agent.id] = prediction
            collective_pred = self.agent.in_weights[self.agent.id] * prediction
            for neigh in self.agent.in_neighbors:
                collective_pred += self.agent.in_weights[neigh] * data[neigh]
            return tf.argmax(collective_pred, axis=1, output_type=tf.int32)

        def test(self, print_log=False):
            prediction = tf.argmax(self.model(self.X_test), axis=1, output_type=tf.int32)
            self.test_accuracy = self.test_metric(prediction, self.Y_test)
            if print_log:
                print(self.agent.id, self.test_accuracy)
            self.test_metric.reset_states()

        def run(
                self, epochs=1, self_learning_epochs=5, review_epochs=1, review_intervals={0: 100},
                weights_computation_interval=100, test_interval=1000, batch_size=10, gamma_weight=100, 
                verbose=False):
            self.accuracies = []
            # self learning
            self.self_learning(epochs=self_learning_epochs)
            self.eval_accuracy()
            self.change_weights(gamma_weight=100)

            review_changes = np.asarray(list(review_intervals.keys()))[::-1]
            Xs_len = len(self.X_shared)
            total_iters = epochs*Xs_len
            for ep in range(epochs):
                # collective learning
                for k in range(Xs_len//batch_size):
                    if verbose:
                        if self.agent.id == 0:
                            print("Remaining {}".format(total_iters-(ep*Xs_len+k*batch_size)), end="\r")
                    # produce collective proxy label
                    samples = self.X_shared[k*batch_size:(k+1)*batch_size]
                    proxy_labels = self.collective_prediction(samples, batch_size)
                    # train on proxy data
                    loss_value, grads = self.grad(tf.reshape(samples, (batch_size, 28, 28, 1)), proxy_labels)
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables), self.stepsize)

                    current_interval = review_changes[np.argmax(review_changes <= k*batch_size)]
                    if (k*batch_size) % review_intervals[current_interval] == 0:
                        self.self_learning(epochs=review_epochs)

                    if (k*batch_size) % weights_computation_interval == 0:
                        self.eval_accuracy()
                        self.change_weights(gamma_weight=gamma_weight)

                    if (k*batch_size) % test_interval == 0:
                        self.test()
                        self.accuracies.append(float(self.test_accuracy.numpy()))

            if self.enable_log:
                return self.accuracies