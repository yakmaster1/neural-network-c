#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "win_ext.h"
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
};

struct NetworkStats
{
	int* neurons_per_layer;
	int layers;
};

void print_activation_vector(NeuralNetwork* network, int index);

#endif