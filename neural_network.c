#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "neural_network.h"
#include "network_logging.h"

NeuralNetwork* init_network(NetworkStats* stats)
{
	NeuralNetwork* network = malloc(sizeof(NeuralNetwork));
	if (network == NULL) {printf("Malloc failed!\n");exit(EXIT_FAILURE);}

	network->stats = stats;

	int layers = network->stats->layers;

	network->activations = malloc(sizeof(Vector*) * layers);
	if (network->activations == NULL) {printf("Malloc failed!\n");exit(EXIT_FAILURE);}
	network->weights = malloc(sizeof(Matrix*) * (layers-1));
	if (network->weights == NULL) {printf("Malloc failed!\n");exit(EXIT_FAILURE);}
	network->biases = malloc(sizeof(Vector*) * (layers-1));
	if (network->biases == NULL) {printf("Malloc failed!\n");exit(EXIT_FAILURE);}
	network->deltas = malloc(sizeof(Vector*) * (layers-1));
	if (network->deltas == NULL) {printf("Malloc failed!\n");exit(EXIT_FAILURE);}
	network->weighted_sums = malloc(sizeof(Vector*) * (layers-1));
	if (network->weighted_sums == NULL) {printf("Malloc failed!\n");exit(EXIT_FAILURE);}
	network->desired_output = createzero_v(network->stats->neurons_per_layer[layers-1]);
	network->error_vector = createzero_v(network->stats->neurons_per_layer[layers-1]);

	for (int i = 0; i < layers; i++)
	{
		int layer_size = network->stats->neurons_per_layer[i];
		network->activations[i] = createzero_v(layer_size);
		if (i > 0)
		{
			network->biases[i-1] = createrandom_v(layer_size);
			network->deltas[i-1] = createzero_v(layer_size);
			network->weighted_sums[i-1] = createzero_v(layer_size);
		}
	}
	for (int i = 0; i < layers -1; i++)
	{
		int prev_layer_size = network->stats->neurons_per_layer[i];
		int layer_size = network->stats->neurons_per_layer[i+1];
		network->weights[i] = createrandom_m(layer_size, prev_layer_size);
	}
	return network;
}

NetworkStats* init_network_stats(int* layer_sizes, int layers)
{
	NetworkStats* stats = malloc(sizeof(NetworkStats));
	if (stats == NULL) {printf("Malloc failed!\n");exit(EXIT_FAILURE);}
	stats->neurons_per_layer = malloc(sizeof(int) * layers);
	if (stats->neurons_per_layer == NULL) {printf("Malloc failed!\n");exit(EXIT_FAILURE);}
	for (int i = 0; i < layers; i++)
	{
		stats->neurons_per_layer[i] = layer_sizes[i];
	}
	stats->layers = layers;
	return stats;
}

void print_network_stats(NeuralNetwork* network)
{
	printf("Number of all layers: %d\n", network->stats->layers);
	for (int i = 0; i < network->stats->layers; i++)
	{
		printf("Neurons in layer %d: %d\n", i+1, network->stats->neurons_per_layer[i]);
	}
}

void set_desired_input(NeuralNetwork* network, Vector* desired_input_vector)
{
	int input_layer_size = network->stats->neurons_per_layer[0];
	if(!(input_layer_size == desired_input_vector->size)) {printf("ERROR => set_desired_input => 1\n"); return;}
	for (int i = 0; i < desired_input_vector->size; i++)
	{
		network->activations[0]->elements[i] = desired_input_vector->elements[i];
	}
	dispose_v(desired_input_vector);
}

void set_desired_output(NeuralNetwork* network, Vector* desired_output_vector)
{
	int index_last_layer = network->stats->layers -1;
	int output_layer_size = network->stats->neurons_per_layer[index_last_layer];
	if(!(output_layer_size == desired_output_vector->size)) {printf("ERROR => set_desired_output => 1\n"); return;}
	for (int i = 0; i < output_layer_size; i++)
	{
		network->desired_output->elements[i] = desired_output_vector->elements[i];
	}
	dispose_v(desired_output_vector);
}

void dispose_network(NeuralNetwork* network)
{
	if (!network) return;
	int network_layers = network->stats->layers;
	dispose_v(network->desired_output);
	dispose_v(network->error_vector);
	for (int i = 0; i < network_layers; i++) {
		dispose_v(network->activations[i]);
	}
	free(network->activations);
	for (int i = 0; i < network_layers -1; i++) {
		dispose_m(network->weights[i]);
		dispose_v(network->biases[i]);
		dispose_v(network->deltas[i]);
		dispose_v(network->weighted_sums[i]);
	}
	free(network->weights);
	free(network->biases);
	free(network->deltas);
	free(network->weighted_sums);

	free(network->stats->neurons_per_layer);
	free(network->stats);
	free(network);
}

void compute_activation(NeuralNetwork* network, int index)
{
	if (index < 1) {printf("ERROR => compute_activation => 1\n"); return;}
	if (index > network->stats->layers -1) {printf("ERROR => compute_activation => 2\n"); return;}
	Matrix* weight_matrix = network->weights[index-1];
	Vector* activation_vector = network->activations[index-1];
	Vector* bias_vector = network->biases[index-1];
	int output_size = network->activations[index]->size;

	Vector* transformed_vector = transform_v(weight_matrix, activation_vector);
	Vector* added = add_v(transformed_vector, bias_vector);

	if (output_size == added->size) 
	{
		for (int i = 0; i < output_size; i++)
		{
			network->activations[index]->elements[i] = sigmoid(added->elements[i]);
			network->weighted_sums[index-1]->elements[i] = added->elements[i];
		}
	}
	else
	{
		printf("Something went horribly wrong. Shutting down...\n");
		exit(EXIT_FAILURE);
	}
	dispose_v(transformed_vector);
	dispose_v(added);
}

void feed_forward(NeuralNetwork* network)
{
	int network_layers = network->stats->layers;
	for (int i = 0; i < network_layers; i++)
	{
		if (!(i == 0))
		{
			compute_activation(network, i);
		}
	}
}

float network_cost(NeuralNetwork* network)
{
	float cost = 0.0f;
	int last_layer_index = network->stats->layers -1;
	for (int i = 0; i < network->desired_output->size; i++)
	{
		network->error_vector->elements[i] = network->activations[last_layer_index]->elements[i] - network->desired_output->elements[i];
		float diff = network->activations[last_layer_index]->elements[i] - network->desired_output->elements[i];
		cost += diff * diff;
	}
	return cost;
}

void backpropagate(NeuralNetwork* network)
{
	int layers = network->stats->layers;
	for (int i = 0; i < layers -1; i++)
	{
		int layer_index = layers-i-1;
		int inner_index = layer_index -1;
		int layer_size = network->stats->neurons_per_layer[layer_index];	
		if (layer_index == layers -1)
		{
			for (int i = 0; i < layer_size; i++)
			{
				float a = network->activations[layer_index]->elements[i];
				float y = network->desired_output->elements[i];
				float z = network->weighted_sums[inner_index]->elements[i];
				float dfz = sigmoid_derivative(z);
				float delta = (a - y) * dfz;
				network->deltas[inner_index]->elements[i] = delta;
			}
		}
		else
		{
			
			Matrix* transposed_weights = transpose_m(network->weights[layer_index]);
			Vector* error_vector = transform_v(transposed_weights, network->deltas[layer_index]);
			for (int i = 0; i < layer_size; i++)
			{
				float z = network->weighted_sums[inner_index]->elements[i];
				float dfz = sigmoid_derivative(z);
				float e = error_vector->elements[i];
				float delta = e * dfz;
				network->deltas[inner_index]->elements[i] = delta;
			}	
			dispose_v(error_vector);
			dispose_m(transposed_weights);
		}
	}
}

void train_network(NeuralNetwork* network, float learning_rate)
{
	for (int layer = 0; layer < network->stats->layers-1; layer++)
	{
		int neurons_per_layer = network->stats->neurons_per_layer[layer+1];
		for (int i = 0; i < neurons_per_layer; i++)
		{
			network->biases[layer]->elements[i] = network->biases[layer]->elements[i] - learning_rate * network->deltas[layer]->elements[i];	
		}
	}
}

int main()
{
	srand(time(NULL));

	NetworkStats* stats = init_network_stats((int[]){10,5,3},3);
	NeuralNetwork* network = init_network(stats);
	
	set_desired_input(network, create_v((float[]){5,4,3,8,1,7,3,8,6,9},10));
	set_desired_output(network, create_v((float[]){0,1,0},3));
	feed_forward(network);

	//print_network_cost(network);
	//print_error_vector(network);

	backpropagate(network);
	train_network(network, 0.1f);

	//print_weight_matrix(network, 0);

	dispose_network(network);
}
