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


typedef struct NeuralLayer NeuralLayer;
typedef struct NeuralNetwork NeuralNetwork;
typedef struct Neuron Neuron;
typedef struct NeuronConnection NeuronConnection;
typedef struct NetworkStats NetworkStats;

struct NeuralLayer
{
	int size;
	Neuron** neurons;
};

struct Neuron 
{
    int num_inputs;
	NeuronConnection** inputs;

	int num_outputs;
	NeuronConnection** outputs;
};

struct NeuronConnection 
{
    Neuron* from;
    Neuron* to;
};

struct NeuralNetwork 
{
	NetworkStats* stats;
	int num_layers;
	NeuralLayer** layers;

	Matrix** weights;
	Vector** biases;
	Vector** activations;
};

struct NetworkStats
{
	int* neurons_per_layer;
	int layers;
	char activation_method;
};

#endif