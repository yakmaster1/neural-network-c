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
	network->biases = malloc(sizeof(Vector*) * layers);
	CHECK_MALLOC(network->biases);
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

NetworkStats* init_network_stats(int* layer_sizes, int layers, char activation_method)
{
	NetworkStats* stats = malloc(sizeof(NetworkStats));
	CHECK_MALLOC(stats);
	stats->neurons_per_layer = malloc(sizeof(int) * layers);
	CHECK_MALLOC(stats->neurons_per_layer);
	stats->activation_method = activation_method;
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
		dispose_v(network->biases[i]);
	}
	free(network->activations);
	free(network->biases);
	
	for (int i = 0; i < network->num_layers - 1; i++) {
		dispose_m(network->weights[i]);
	}
	free(network->weights);
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

void compute_activation(NeuralNetwork* network, int layer)
{
	if (layer <= 1) {printf("ERROR => compute_activation => 1\n"); return;}
	if (layer > network->num_layers) {printf("ERROR => compute_activation => 2\n"); return;}
	int index = layer-1;
	Matrix* weight_matrix = network->weights[index-1];
	Vector* activation_vector = network->activations[index-1];
	Vector* bias_vector = network->biases[index];
	Vector* output_activations = network->activations[index];

	Vector* transformed_vector = transform_v(weight_matrix, activation_vector);
	Vector* added = add_v(transformed_vector, bias_vector);
	char method = network->stats->activation_method;
	if (method == 'S' || method == 's') {sigmoid_v(added);}
	else 
	{
		printf("There is no activation function with the key '%c'. Using fallback 'S'\n", method);
		sigmoid_v(added);
	}

	if (output_activations->size == added->size) 
	{
		for (int i = 0; i < output_activations->size; i++)
		{
			output_activations->elements[i] = added->elements[i];
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
	if(!(network->layers[0]->size == input_vector->size)) {printf("ERROR => compute_for_input => 1"); return NULL;}
	for (int i = 0; i < input_vector->size; i++)
	{
		network->activations[0]->elements[i] = input_vector->elements[i];
	}
	int network_layers = network->stats->layers;
	for (int i = 0; i < network_layers-1; i++)
	{
		compute_activation(network, i+2);
	}
	Vector* desired_output = create_single_number_v(input_vector->size, output_neuron_index, 1.0f);
	CHECK_MALLOC(desired_output);
	int size_last_layer = network->layers[network_layers-1]->size;
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

void print_weight_matrix(NeuralNetwork* network, int layer)
{
	if (layer < 0) {printf("ERROR => print_weight_matrix => 1"); return;}
	if (layer >= network->num_layers-1) {printf("ERROR => print_weight_matrix => 2\n"); return;}
	print_m(network->weights[layer]);
}

void print_activation_vector(NeuralNetwork* network, int layer)
{
	if (layer < 0) {printf("ERROR => print_activation_vector => 1\n"); return;}
	if (layer >= network->num_layers) {printf("ERROR => print_activation_vector => 2\n"); return;}
	print_v(network->activations[layer]);
}

void print_bias_vector(NeuralNetwork* network, int layer)
{
	if (layer < 0) {printf("ERROR => print_bias_vector => 1\n"); return;}
	if (layer >= network->num_layers) {printf("ERROR => print_bias_vector => 2\n"); return;}
	if(layer == 0) {printf("NOTE: The bias of the first layer is not used.\n");}
	print_v(network->biases[layer]);
}

float calculate_cost(Vector* network_output, Vector* desired_output)
{
	float cost = 0.0f;
	if (network_output->size != desired_output->size)
	{
		printf("Something went horribly wrong. Shutting down...\n");
		exit(EXIT_FAILURE);
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
		(int[]){10,5,5},
		3,
		'S'
	);
	NeuralNetwork* network = init_network(stats);
	
	Vector* input = create_v((float[]){5,4,3,8,1,7,3,8,6,9}, 10);
	Vector* v = compute_for_input(network, input, 3);
	print_v(v);

	dispose_network(network);
}
