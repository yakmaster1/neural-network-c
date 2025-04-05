#ifndef NETWORK_LOGGING
#define NETWORK_LOGGING

#include "neural_network.h"

void print_activation_vector(NeuralNetwork* network, int index);
void print_output(NeuralNetwork* network);
void print_bias_vector(NeuralNetwork* network, int index);
void print_delta_vector(NeuralNetwork* network, int index);
void print_weighted_sum_vector(NeuralNetwork* network, int index);
void print_weight_matrix(NeuralNetwork* network, int index);
void print_error_vector(NeuralNetwork* network);
void print_network_cost(NeuralNetwork* network);

#endif