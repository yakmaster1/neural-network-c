#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include "algebra.h"

#define INPUT_LAYER 28*28
#define HIDDEN_L1 16
#define HIDDEN_L2 16
#define OUTPUT_LAYER 10

typedef struct NeuralNetwork NeuralNetwork;
typedef struct NetworkStats NetworkStats;

struct NeuralNetwork 
{
	NetworkStats* stats;

	Matrix** weights;
	Vector** biases;
	Vector** activations;
	Vector** deltas;
	Vector** weighted_sums;

	Vector* desired_output;
	Vector* error_vector;
};

struct NetworkStats
{
	int* neurons_per_layer;
	int layers;
};

float network_cost(NeuralNetwork* network);

#endif