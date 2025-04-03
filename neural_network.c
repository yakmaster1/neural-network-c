#include "neural_network.h"

NeuronConnection* init_connection(Neuron* from, Neuron* to)
{
	NeuronConnection* connection = malloc(sizeof(NeuronConnection));
	connection->from = from;
	connection->to = to;
	return connection;
}

Neuron* init_neuron(int num_inputs, int num_outputs)
{
	Neuron* neuron = malloc(sizeof(Neuron));
	neuron->num_inputs = num_inputs;
	neuron->num_outputs = num_outputs;
	neuron->inputs = malloc(sizeof(NeuronConnection*) * num_inputs);
	neuron->outputs = malloc(sizeof(NeuronConnection*) * num_outputs);
	return neuron;
}

NeuralLayer* init_layer(NeuralNetwork* network, int size, int index, int num_inputs, int num_outputs)
{
	NeuralLayer* layer = malloc(sizeof(NeuralLayer));
	layer->size = size;
	layer->neurons = malloc(sizeof(Neuron*) * size);
	network->activations[index] = createzero_v(size);
	network->biases[index] = createzero_v(size);
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
	int layers = stats->layers;
	int* layer_sizes = stats->neurons_per_layer;
	network->num_layers = layers;
	network->layers = malloc(sizeof(NeuralLayer*) * layers);
	network->activations = malloc(sizeof(Vector*) * layers);
	network->weights = malloc(sizeof(Matrix*) * (layers-1));
	network->biases = malloc(sizeof(Vector*) * layers);
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
	stats->neurons_per_layer = malloc(sizeof(int) * layers);
	for (int i = 0; i < layers; i++)
	{
		stats->neurons_per_layer[i] = layer_sizes[i];
	}
	stats->layers = layers;
	return stats;
}

void dispose_network(NeuralNetwork* network, NetworkStats* stats)
{
	if (!network) return;
	for (int i = 0; i < network->num_layers; i++) {
		dispose_v(network->activations[i]);
		dispose_v(network->biases[i]);
	}
	free(network->activations);
	free(network->biases);
	
	for (int i = 0; i < network->num_layers - 1; i++) {
		dispose_m(network->weights[i]);
	}
	free(network->weights);
	for (int l = 0; network->layers[l] != NULL; l++)
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
	free(network);
	free(stats->neurons_per_layer);
	free(stats);
}

void compute_activation(NeuralNetwork* network, int layer)
{
	if (layer < 1) 
	{
		printf("[ERROR] compute_activation() -> layer < 1: %d < 1\n", layer);
		return;
	}
	if (layer >= network->num_layers)
	{
		printf("[ERROR] compute_activation() -> layer >= network->num_layers: %d >= %d\n", layer, network->num_layers);
		return;		
	}
	Matrix* weight_matrix = network->weights[layer-1];
	Vector* activation_vector = network->activations[layer-1];
	Vector* bias_vector = network->biases[layer];
	Vector* output_activations = network->activations[layer];

	Vector* transformation = transform_v(weight_matrix, activation_vector);
	Vector* added = add_v(transformation, bias_vector);
	sigmoid_v(added);

	if (output_activations->size == added->size) 
	{
		for (int i = 0; i < output_activations->size; i++)
		{
			output_activations->elements[i] = added->elements[i];
		}
	}
	else
	{
		printf("Something went horribly wrong. Please shut down the nework immediately!\n");
	}

	dispose_v(transformation);
	dispose_v(added);
}

float calculate_cost(Vector* network_output, Vector* desired_output)
{
	float cost = 0.0f;
	if (network_output->size != desired_output->size)
	{
		printf("Something went horribly wrong. Please shut down the nework immediately!\n");
		return cost;
	}
	for (int i = 0; i < network_output->size; i++)
	{
		cost += exp2(network_output->elements[i] - desired_output->elements[i]);
	}
	return cost;
}

void print_network_stats(NetworkStats* stats)
{
	//clear_console();
}

int main()
{
	NetworkStats* stats = init_network_stats(
		(int[]){10,5,1},
		3
	);
	NeuralNetwork* network = init_network(stats);
	
	compute_activation(network, 1);

	dispose_network(network, stats);
}
