#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "alglib.h"

typedef struct NeuralNetwork NeuralNetwork;

struct NeuralNetwork
{
	char identifier;
    int size;
    int *neurons_per_layer;
    float cost;
    
    Matrix **weights;
	Vector **biases;
	Vector **activations;
	Vector **deltas;
	Vector **weighted_sums;

	Vector *desired_output;
};

void set_network_input(NeuralNetwork *network, Vector *input_vector);
void compute_activation(NeuralNetwork *network);
void print_draw_output(NeuralNetwork *network);

#endif