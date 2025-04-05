#include "neural_network.h"

NeuralNetwork* init_network(NetworkStats* stats)
{
	NeuralNetwork* network = malloc(sizeof(NeuralNetwork));
	CHECK_MALLOC(network);

	network->stats = stats;

	int layers = network->stats->layers;

	network->activations = malloc(sizeof(Vector*) * layers);
	CHECK_MALLOC(network->activations);
	network->weights = malloc(sizeof(Matrix*) * (layers-1));
	CHECK_MALLOC(network->weights);
	network->biases = malloc(sizeof(Vector*) * (layers-1));
	CHECK_MALLOC(network->biases);
	network->deltas = malloc(sizeof(Vector*) * (layers-1));
	CHECK_MALLOC(network->deltas);
	network->weighted_sums = malloc(sizeof(Vector*) * (layers-1));
	CHECK_MALLOC(network->weighted_sums);

	for (int i = 0; i < layers; i++)
	{
		int layer_size = network->stats->neurons_per_layer[i];
		network->activations[i] = createzero_v(layer_size);
		if (i > 0)
		{
			network->biases[i-1] = createzero_v(layer_size);
			network->deltas[i-1] = createzero_v(layer_size);
			network->weighted_sums[i-1] = createzero_v(layer_size);
		}
	}
	for (int i = 0; i < layers -1; i++)
	{
		int prev_layer_size = network->stats->neurons_per_layer[i];
		int next_layer_size = network->stats->neurons_per_layer[i+1];
		network->weights[i] = create_m(next_layer_size, prev_layer_size);
	}
	return network;
}

NetworkStats* init_network_stats(int* layer_sizes, int layers)
{
	NetworkStats* stats = malloc(sizeof(NetworkStats));
	CHECK_MALLOC(stats);
	stats->neurons_per_layer = malloc(sizeof(int) * layers);
	CHECK_MALLOC(stats->neurons_per_layer);
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

void print_network_elements(NeuralNetwork* network)
{
	printf("Activations | Biases | Deltas | Weighted sums\n");
	for (int i = 0; i < network->stats->layers; i++)
	{
		int columns = (i == 0) ? 1 : 4;
		Matrix* matrix = create_m(network->stats->neurons_per_layer[i], columns);
		addcv_m(matrix, copy_v(network->activations[i]), 0);
		if (i < 0)
		{
			addcv_m(matrix, copy_v(network->biases[i]), 1);
			addcv_m(matrix, copy_v(network->deltas[i]), 2);
			addcv_m(matrix, copy_v(network->weighted_sums[i]), 3);			
		}
		print_m(matrix);
		printf("\n");
		free(matrix);
	}
	
}

void dispose_network(NeuralNetwork* network)
{
	if (!network) return;
	int network_layers = network->stats->layers;
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

Vector* compute_for_input(NeuralNetwork* network, Vector* input_vector, int output_neuron_index)
{
	int input_layer_size = network->stats->neurons_per_layer[0];
	if(!(input_layer_size == input_vector->size)) {printf("ERROR => compute_for_input => 1\n"); return NULL;}
	for (int i = 0; i < input_vector->size; i++)
	{
		network->activations[0]->elements[i] = input_vector->elements[i];
	}
	int network_layers = network->stats->layers;
	for (int i = 0; i < network_layers; i++)
	{
		if (!(i == 0))
		{
			compute_activation(network, i);
		}
	}
	int size_last_layer = network->stats->neurons_per_layer[network_layers-1];
	Vector* desired_output = create_single_number_v(size_last_layer, output_neuron_index, 1.0f);
	CHECK_MALLOC(desired_output);
	float* cost_vector_elements = calloc(size_last_layer, sizeof(float));
	CHECK_MALLOC(cost_vector_elements);
	for (int i = 0; i < size_last_layer; i++)
	{
		cost_vector_elements[i] = network->activations[network_layers-1]->elements[i] - desired_output->elements[i];;
	}
	Vector* cost_vector = create_v(cost_vector_elements, size_last_layer);
	dispose_v(desired_output);
	free(cost_vector_elements);
	return cost_vector;
}

void print_activation_vector(NeuralNetwork* network, int index)
{
	if (index < 0) {printf("ERROR => print_activation_vector => 1\n"); return;}
	if (index > network->stats->layers -1) {printf("ERROR => print_activation_vector => 2\n"); return;}
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

int main()
{
	NetworkStats* stats = init_network_stats(
		(int[]){10,5,3},
		3
	);

	NeuralNetwork* network = init_network(stats);
	
	Vector* input = create_v((float[]){5,4,3,8,1,7,3,8,6,9}, 10);
	Vector* v = compute_for_input(network, input, 2);
	print_v(v);
	
	print_network_elements(network);

	dispose_network(network);
}
