#include "neural_network.h"

NeuronConnection* init_connection(Neuron* from, Neuron* to)
{
	NeuronConnection* connection = malloc(sizeof(NeuronConnection));
	CHECK_MALLOC(connection);
	connection->from = from;
	connection->to = to;
	return connection;
}

Neuron* init_neuron(int num_inputs, int num_outputs)
{
	Neuron* neuron = malloc(sizeof(Neuron));
	CHECK_MALLOC(neuron);
	neuron->num_inputs = num_inputs;
	neuron->num_outputs = num_outputs;
	neuron->inputs = malloc(sizeof(NeuronConnection*) * num_inputs);
	CHECK_MALLOC(neuron->inputs);
	neuron->outputs = malloc(sizeof(NeuronConnection*) * num_outputs);
	CHECK_MALLOC(neuron->outputs);
	return neuron;
}

NeuralLayer* init_layer(NeuralNetwork* network, int size, int index, int num_inputs, int num_outputs)
{
	NeuralLayer* layer = malloc(sizeof(NeuralLayer));
	CHECK_MALLOC(layer);
	layer->size = size;
	layer->neurons = malloc(sizeof(Neuron*) * size);
	CHECK_MALLOC(layer->neurons);
	network->activations[index] = createzero_v(size);
	if (index >= network->num_layers -2)
	{
		network->biases[index] = createzero_v(size);
		network->deltas[index] = createzero_v(size);
		network->weighted_sums[index] = createzero_v(size);
	}
	for (int i = 0; i < size; i++)
	{
		layer->neurons[i] = init_neuron(num_inputs, num_outputs);
	}
	return layer;
}

void connect_layers(NeuralNetwork* network, int start_layer, NetworkStats* stats)
{
	NeuralLayer* from = network->layers[start_layer];
	NeuralLayer* to = network->layers[start_layer+1];
	int matrix_columns = stats->neurons_per_layer[start_layer];
	int matrix_rows = stats->neurons_per_layer[start_layer+1];
	network->weights[start_layer] = create_m(matrix_rows, matrix_columns);

	for (int i = 0; i < from->size; i++) {
		Neuron* n_from = from->neurons[i];
		for (int j = 0; j < to->size; j++) {
			Neuron* n_to = to->neurons[j];
			NeuronConnection* conn = init_connection(n_from, n_to);
			n_from->outputs[j] = conn;
			n_to->inputs[i] = conn;
		}
	}
}

NeuralNetwork* init_network(NetworkStats* stats)
{
	NeuralNetwork* network = malloc(sizeof(NeuralNetwork));
	CHECK_MALLOC(network);
	int layers = stats->layers;
	int* layer_sizes = stats->neurons_per_layer;
	network->num_layers = layers;
	network->layers = malloc(sizeof(NeuralLayer*) * layers);
	CHECK_MALLOC(network->layers);
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
	network->stats = stats;
	for (int i = 0; i < layers; i++) {
		int num_inputs = (i == 0) ? 0 : layer_sizes[i - 1];
		int num_outputs = (i == layers - 1) ? 0 : layer_sizes[i + 1];
		network->layers[i] = init_layer(network, layer_sizes[i], i, num_inputs, num_outputs);
	}
	for (int i = 0; i < layers - 1; i++) {
		connect_layers(network, i, stats);
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

void dispose_network(NeuralNetwork* network)
{
	if (!network) return;
	for (int i = 0; i < network->num_layers; i++) {
		dispose_v(network->activations[i]);
	}
	free(network->activations);
	for (int i = 0; i < network->num_layers - 2; i++) {
		dispose_m(network->weights[i]);
	}
	free(network->weights);
	for (int i = 0; i < network->num_layers - 1; i++)
	{
		dispose_v(network->biases[i]);
		dispose_v(network->deltas[i]);
		dispose_v(network->weighted_sums[i]);
	}
	free(network->biases);
	free(network->deltas);
	free(network->weighted_sums);
	for (int l = 0; l < network->num_layers; l++)
	{
		NeuralLayer* layer = network->layers[l];
		for (int n = 0; n < layer->size; n++)
		{
			Neuron* neuron = layer->neurons[n];
			for (int o = 0; o < neuron->num_outputs; o++) {
				free(neuron->outputs[o]);
			}
			free(neuron->inputs);
			free(neuron->outputs);
			free(neuron);
		}
		free(layer->neurons);
		free(layer);
	}
	free(network->layers);
	free(network->stats->neurons_per_layer);
	free(network->stats);
	free(network);
}

void compute_activation(NeuralNetwork* network, int index)
{
	if (index < 1) {printf("ERROR => compute_activation => 1\n"); return;}
	if (index > network->num_layers -1) {printf("ERROR => compute_activation => 2\n"); return;}
	Matrix* weight_matrix = network->weights[index-1];
	Vector* activation_vector = network->activations[index-1];
	Vector* bias_vector = network->biases[index];
	int output_size = network->activations[index]->size;

	Vector* transformed_vector = transform_v(weight_matrix, activation_vector);
	Vector* added = add_v(transformed_vector, bias_vector);

	if (output_size == added->size) 
	{
		for (int i = 0; i < output_size; i++)
		{
			network->activations[index]->elements[i] = sigmoid(added->elements[i]);
			network->weighted_sums[index]->elements[i] = added->elements[i];
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
	if(!(network->layers[0]->size == input_vector->size)) {printf("ERROR => compute_for_input => 1\n"); return NULL;}
	for (int i = 0; i < input_vector->size; i++)
	{
		network->activations[0]->elements[i] = input_vector->elements[i];
	}
	int network_layers = network->stats->layers;
	for (int i = 1; i < network_layers-1; i++)
	{
		compute_activation(network, i);
	}
	int size_last_layer = network->layers[network_layers-1]->size;
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

void print_weight_matrix(NeuralNetwork* network, int index)
{
	if (index < 0) {printf("ERROR => print_weight_matrix => 1"); return;}
	if (index >= network->num_layers-1) {printf("ERROR => print_weight_matrix => 2\n"); return;}
	print_m(network->weights[index]);
}

void print_activation_vector(NeuralNetwork* network, int index)
{
	if (index < 0) {printf("ERROR => print_activation_vector => 1\n"); return;}
	if (index > network->num_layers-1) {printf("ERROR => print_activation_vector => 2\n"); return;}
	print_v(network->activations[index]);
}

void print_bias_vector(NeuralNetwork* network, int index)
{
	if (index < 1) {printf("ERROR => print_bias_vector => 1\n"); return;}
	if (index > network->num_layers -1) {printf("ERROR => print_bias_vector => 2\n"); return;}
	print_v(network->biases[index]);
}

void print_delta_vector(NeuralNetwork* network, int index)
{
	if (index < 1) {printf("ERROR => print_delta_vector => 1\n"); return;}
	if (index > network->num_layers -1) {printf("ERROR => print_delta_vector => 2\n"); return;}
	print_v(network->deltas[index]);
}

void print_weighted_sum_vector(NeuralNetwork* network, int index)
{
	if (index < 1) {printf("ERROR => print_weighted_sum_vector => 1\n"); return;}
	if (index > network->num_layers -1) {printf("ERROR => print_weighted_sum_vector => 2\n"); return;}
	print_v(network->weighted_sums[index]);
}

void print_network_stats(NetworkStats* stats)
{
	//clear_console();
}

int main()
{
	NetworkStats* stats = init_network_stats(
		(int[]){10,5,3},
		3
	);
	NeuralNetwork* network = init_network(stats);
	
	Vector* input = create_v((float[]){5,4,3,8,1,7,3,8,6,9}, 10);
	Vector* v = compute_for_input(network, input, 3);
	print_v(v);
	printf("\n");
	print_weight_matrix(network, 0);

	dispose_network(network);
}
