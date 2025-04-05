#include <stdio.h>

#include "network_logging.h"
#include "algebra.h"

void print_activation_vector(NeuralNetwork* network, int index)
{
	if (index < 0) {printf("ERROR => print_activation_vector => 1\n"); return;}
	if (index > network->stats->layers -1) {printf("ERROR => print_activation_vector => 2\n"); return;}
	print_v(network->activations[index]);
}

void print_output(NeuralNetwork* network)
{
	int index = network->stats->layers -1;
	print_v(network->activations[index]);
}

void print_bias_vector(NeuralNetwork* network, int index)
{
	if (index < 0) {printf("ERROR => print_bias_vector => 1\n"); return;}
	if (index > network->stats->layers -2) {printf("ERROR => print_bias_vector => 2\n"); return;}
	print_v(network->biases[index]);
}

void print_delta_vector(NeuralNetwork* network, int index)
{
	if (index < 0) {printf("ERROR => print_delta_vector => 1\n"); return;}
	if (index > network->stats->layers -2) {printf("ERROR => print_delta_vector => 2\n"); return;}
	print_v(network->deltas[index]);
}

void print_weighted_sum_vector(NeuralNetwork* network, int index)
{
	if (index < 0) {printf("ERROR => print_weighted_sum_vector => 1\n"); return;}
	if (index > network->stats->layers -2) {printf("ERROR => print_weighted_sum_vector => 2\n"); return;}
	print_v(network->weighted_sums[index]);
}

void print_weight_matrix(NeuralNetwork* network, int index)
{
	if (index < 0) {printf("ERROR => print_weight_matrix => 1\n"); return;}
	if (index > network->stats->layers -2) {printf("ERROR => print_weight_matrix => 2\n"); return;}
	print_m(network->weights[index]);
}

void print_error_vector(NeuralNetwork* network)
{
    print_v(network->error_vector);
}

void print_network_cost(NeuralNetwork* network)
{
	printf("Cost: %.2f\n", network_cost(network));
}