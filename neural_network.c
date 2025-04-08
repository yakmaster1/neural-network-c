#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <windows.h>

#include "alglib.h"

typedef struct NeuralNetwork NeuralNetwork;

struct NeuralNetwork
{
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

NeuralNetwork *init_network(int *neurons_per_layer, int size)
{
    if (size < 2) {printf("Cant create network. Exiting...\n"); exit(EXIT_FAILURE);}
    for (int i = 0; i < size; i++)
    {if (neurons_per_layer[i] < 1) {printf("Cant create network. Exiting...\n"); exit(EXIT_FAILURE);}}
    NeuralNetwork *network = malloc(sizeof(NeuralNetwork));
    if (network == NULL) {printf("Mem alloc failed\n"); return NULL;}
    network->size = size;
    network->cost = 0.0f;
    network->neurons_per_layer = malloc(sizeof(int) * size);
    if (network->neurons_per_layer == NULL) {printf("Mem alloc failed\n"); return NULL;}
    for (int i = 0; i < size; i++) {network->neurons_per_layer[i] = neurons_per_layer[i];}
    network->weights = malloc(sizeof(Matrix*) * (size - 1));
    if (network->weights == NULL) {printf("Mem alloc failed\n"); return NULL;}
    network->biases = malloc(sizeof(Vector*) * (size - 1));
    if (network->biases == NULL) {printf("Mem alloc failed\n"); return NULL;}
    network->activations = malloc(sizeof(Vector*) * size);
    if (network->activations == NULL) {printf("Mem alloc failed\n"); return NULL;}
    network->deltas = malloc(sizeof(Vector*) * (size - 1));
    if (network->deltas == NULL) {printf("Mem alloc failed\n"); return NULL;}
    network->weighted_sums = malloc(sizeof(Vector*) * (size - 1));
    if (network->weighted_sums == NULL) {printf("Mem alloc failed\n"); return NULL;}
    network->desired_output = create_v(network->neurons_per_layer[size -1], (float[]){0}, ZERO);
    srand(time(NULL));
    for (int i = 0; i < size; i++)
    {
        int neurons_layer = network->neurons_per_layer[i];
        network->activations[i] = create_v(neurons_layer, (float[]){0}, ZERO);
    }
    for (int i = 0; i < size -1; i++)
    {
        int neurons_prev_layer = network->neurons_per_layer[i];
        int neurons_layer = network->neurons_per_layer[i+1];
        network->weights[i] = init_matrix_xavier(neurons_layer, neurons_prev_layer); //RAND
        network->biases[i] = create_v(neurons_layer, (float[]){0}, ZERO); //RAND
        network->deltas[i] = create_v(neurons_layer, (float[]){0}, ZERO);
        network->weighted_sums[i] = create_v(neurons_layer, (float[]){0}, ZERO);
    }  
    return network;
}

void dispose_network(NeuralNetwork *network)
{
    int size = network->size;
    for (int i = 0; i < size; i++) {dispose_v(network->activations[i]);}
    for (int i = 0; i < size -1; i++) {
        dispose_m(network->weights[i]);
        dispose_v(network->biases[i]);
        dispose_v(network->deltas[i]);
        dispose_v(network->weighted_sums[i]);
    }
    dispose_v(network->desired_output);
    free(network->weighted_sums);
    free(network->deltas);
    free(network->biases);
    free(network->weights);
    free(network->activations);
    free(network->neurons_per_layer);
    free(network);
}

void print_network_output(NeuralNetwork *network)
{
    int last_layer_index = network->size -1;
    printf("-----------------------------------------------------\n");
    printf("\nNetwork output:     ");
    for (int i = 0; i < network->neurons_per_layer[last_layer_index]; i++) 
    {
        printf("%.2f [%d]     ", network->activations[last_layer_index]->elements[i], i);
    }
    printf("\nDesired output:     ");
    for (int i = 0; i < network->neurons_per_layer[last_layer_index]; i++) 
    {
        printf("%.2f [%d]     ", network->desired_output->elements[i], i);
    }
    printf("\n\nCost of this iteration: %.2f", network->cost);
}

void print_network_vectors(NeuralNetwork *network, bool extended_stats)
{
    printf("---------------------- Layer 1 ----------------------\n\n");
    print_v(network->activations[0]);
    printf("\n\n");
    for (int i = 0; i < network->size -1; i++)
    {
        printf("---------------------- Layer %d ----------------------\n\n", i+2);
        Vector *vectors[] = {network->activations[i+1], network->biases[i], network->weighted_sums[i], network->deltas[i]};
        printmultiple_v(4, vectors, 2, true);
        printf("\n\n");
    } 
    if (extended_stats == true) {print_network_output(network);}
}

void print_network_weights(NeuralNetwork *network, bool extended_stats)
{
    for (int i = 0; i < network->size -1; i++)
    {
        printf("------------------- Layer %d -> %d -------------------\n\n", i+1, i+2);
        print_m(network->weights[i]);
        printf("\n\n");
    }
    if (extended_stats == true) {print_network_output(network);}
}

void apply_activation_function(NeuralNetwork *network, int layer_index)
{
    if (layer_index > network->size -1) {printf("apply_activation_function -> 1\n"); return;}
    for (int i = 0; i < network->neurons_per_layer[layer_index]; i++)
    {network->activations[layer_index]->elements[i] = sigmoid(network->weighted_sums[layer_index-1]->elements[i]);}
}

void calculate_cost(NeuralNetwork *network)
{
    int index_last_layer = network->size -1;
    float cost = 0.0f;
    for (int i = 0; i < network->neurons_per_layer[index_last_layer]; i++)
    {
        float diff = network->activations[index_last_layer]->elements[i] - network->desired_output->elements[i];
        cost += diff * diff;
    }
    network->cost = 0.5f * cost;
}

void compute_activation(NeuralNetwork *network)
{
    for (int i = 0; i < network->size -1; i++) 
    {
        transform_linear(network->weights[i], network->activations[i], network->weighted_sums[i]);
        for (int j = 0; j < network->neurons_per_layer[i+1]; j++) {
            network->weighted_sums[i]->elements[j] += network->biases[i]->elements[j];
        }
        apply_activation_function(network, i+1);
    }
    calculate_cost(network);
}

void set_network_input(NeuralNetwork *network, Vector *input_vector)
{
    if (input_vector->size != network->neurons_per_layer[0]) {printf("set_network_input -> 1\n"); return;}
    for (int i = 0; i < input_vector->size; i++)
    {network->activations[0]->elements[i] = input_vector->elements[i];}
}

void set_desired_single_output(NeuralNetwork *network, int output_index)
{
    if (output_index > network->desired_output->size -1) {printf("set_desired_single_output -> 1\n"); return;}
    int index_last_layer = network->size -1;
    for (int i = 0; i < network->desired_output->size; i++)
    {
        if (output_index == i) {network->desired_output->elements[i] = 1;}
        else {network->desired_output->elements[i] = 0;}
    }
}

void train_network(NeuralNetwork *network, float learning_rate)
{
    for (int ns = 0; ns < network->size - 1; ns++)
    {
        int layer_size = network->neurons_per_layer[ns+1];
        for (int ls = 0; ls < layer_size; ls++)
        {
            network->biases[ns]->elements[ls] -= learning_rate * network->deltas[ns]->elements[ls];
        }

        Matrix *weight_matrix = network->weights[ns];
        Vector *activations = network->activations[ns];
        Vector *deltas = network->deltas[ns];
        for (int c = 0; c < weight_matrix->columns; c++)
        {
            for (int r = 0; r < weight_matrix->rows; r++)
            {
                float weight_value = weight_matrix->vectors[c]->elements[r];
                weight_matrix->vectors[c]->elements[r] -= learning_rate * activations->elements[c] * deltas->elements[r];
            } 
        }
    }   
}

void backpropagate(NeuralNetwork *network, float learning_rate)
{
    int index_last_layer = network->size - 1;
    for (int ns = 0; ns < network->size - 1; ns++)
    {
        int i_desc = index_last_layer - ns;
        int layer_size = network->neurons_per_layer[i_desc];
        for (int ls = 0; ls < layer_size; ls++)
        {
            float abl_act = abl_sigmoid(network->weighted_sums[i_desc -1]->elements[ls]);
            if (i_desc == index_last_layer)
            {
                float network_output = network->activations[i_desc]->elements[ls];
                float desired_output = network->desired_output->elements[ls];
                float value = abl_act * (network_output - desired_output);
                network->deltas[i_desc - 1]->elements[ls] = value;
            }
            else
            {
                float sum = 0.0f;
                for (int pl = 0; pl < network->neurons_per_layer[i_desc +1]; pl++)
                {
                    float weight = network->weights[i_desc]->vectors[ls]->elements[pl];
                    float delta_next = network->deltas[i_desc]->elements[pl];
                    sum += weight * delta_next;
                }
                network->deltas[i_desc - 1]->elements[ls] = abl_act * sum;
            }
        }
    }
    train_network(network, learning_rate);
}

int main()
{
    NeuralNetwork *network = init_network((int[]){6,5,5,3},4);
    Vector *input = create_v(6, (float[]){1,2,3,4,5,6}, INIT);

    set_network_input(network, input);
    set_desired_single_output(network, 2);

    for (int i = 0; i < 200; i++)
    {
        compute_activation(network);
        backpropagate(network, 0.1f);
        system("cls");
        printf("Iteration %d/200\n", i);
        print_network_output(network);
        Sleep(1);
    }

    system("cls");
    print_network_weights(network, false);
    print_network_vectors(network, true);
    
    dispose_network(network);
    network = NULL;

    dispose_v(input);
    input = NULL;
    return 0;
}