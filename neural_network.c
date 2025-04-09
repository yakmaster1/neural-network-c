#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <windows.h>

#include "image_extr.h"
#include "neural_network.h"
#include "draw_window.h"

#define MAX_COMMAND_LENGTH 256

NeuralNetwork *init_network(int *neurons_per_layer, int size, char identifier)
{
    if (size < 2) {printf("Cant create network. Exiting...\n"); exit(EXIT_FAILURE);}
    for (int i = 0; i < size; i++)
    {if (neurons_per_layer[i] < 1) {printf("Cant create network. Exiting...\n"); exit(EXIT_FAILURE);}}
    NeuralNetwork *network = malloc(sizeof(NeuralNetwork));
    if (network == NULL) {printf("Mem alloc failed\n"); return NULL;}
    network->identifier = identifier;
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
        network->weights[i] = pre_init_matrix(neurons_layer, neurons_prev_layer); //RAND
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

void print_draw_output(NeuralNetwork *network)
{
    int last_layer_index = network->size -1;
    int index = 0;
    float value = 0.0f;
    system("cls");
    for (int i = 0; i < network->neurons_per_layer[last_layer_index]; i++) 
    {
        if (value < network->activations[last_layer_index]->elements[i]) 
        {
            value = network->activations[last_layer_index]->elements[i];
            index = i;
        };
        printf("%.2f [%d]\n", network->activations[last_layer_index]->elements[i], i);
    }
    printf("\n\nIm thinking of %d with %.2f%% certainty\n", index, value);
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

float activation_function(float value)
{
    //return sigmoid(value);
    return relu(value);
}

float abl_activation_function(float value)
{
    //return abl_sigmoid(value);
    return abl_relu(value);
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
    {network->activations[layer_index]->elements[i] = activation_function(network->weighted_sums[layer_index-1]->elements[i]);}
}

void calculate_cost_mse(NeuralNetwork *network)
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

void calculate_cost_ce(NeuralNetwork *network)
{
    int index_last_layer = network->size -1;
    float cost = 0.0f;
    for (int i = 0; i < network->neurons_per_layer[index_last_layer]; i++)
    {
        float a = network->activations[index_last_layer]->elements[i];
        float y = network->desired_output->elements[i];
        if (y == 1) cost -= logf(a + 1e-7f);
    }
    network->cost = cost;
}

void softmax(Vector *z, Vector *output)
{
    float max_val = z->elements[0];
    for (int i = 1; i < z->size; i++) 
    {
        if (z->elements[i] > max_val) {max_val = z->elements[i];}
    }
    float sum = 0.0f;
    for (int i = 0; i < z->size; i++) 
    {
        output->elements[i] = expf(z->elements[i] - max_val);
        sum += output->elements[i];
    }
    for (int i = 0; i < z->size; i++) {output->elements[i] /= sum;}
}

void compute_activation(NeuralNetwork *network)
{
    for (int i = 0; i < network->size -1; i++) 
    {
        transform_linear(network->weights[i], network->activations[i], network->weighted_sums[i]);
        for (int j = 0; j < network->neurons_per_layer[i+1]; j++) {
            network->weighted_sums[i]->elements[j] += network->biases[i]->elements[j];
        }
        if (i == network->size -2)
        {
            int index_last_layer = network->size -1;
            softmax(network->weighted_sums[index_last_layer-1], network->activations[index_last_layer]);
        }
        else {apply_activation_function(network, i+1);}
    }
    calculate_cost_ce(network);
}

void set_network_input(NeuralNetwork *network, Vector *input_vector)
{
    if (input_vector->size != network->neurons_per_layer[0]) {printf("set_network_input -> 1\n"); return;}
    for (int i = 0; i < input_vector->size; i++)
    {network->activations[0]->elements[i] = input_vector->elements[i] / 255.0f;} // Normalized
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
            float abl_act = abl_activation_function(network->weighted_sums[i_desc -1]->elements[ls]);
            if (i_desc == index_last_layer)
            {
                float network_output = network->activations[i_desc]->elements[ls];
                float desired_output = network->desired_output->elements[ls];
                //float value = abl_act * (network_output - desired_output); // MSE Sigmoid
                float value = (network_output - desired_output); // CE ReLU
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

void train_network_batch(NeuralNetwork *network, Vector *input, float *input_array, int *label, int *offset, int count, float learning_rate)
{
    int pre_save_offset = 0;
    const int total_images = 60000;

    for (int i = 0; i < count; i++)
    {
        int data_index = (i + *offset) % total_images;
        get_input_data(input_array, data_index + *offset, label);
        setvalues_v(input, input_array, IMAGE_SIZE);
        set_network_input(network, input);
        set_desired_single_output(network, *label);
        
        compute_activation(network);
        backpropagate(network, learning_rate);

        if ((i + *offset) % 50 == 0) {
            system("cls");
            printf("Iteration %d/60.000\n", i + *offset);
            print_network_output(network);
        }

        pre_save_offset++;
    }
    *offset = (*offset + pre_save_offset) % total_images;
}

void save_network_data(NeuralNetwork *network)
{
    char filename[256];
    sprintf(filename, "data/%c.txt", network->identifier);
    
    FILE *f = fopen(filename, "w");
    if (f == NULL) 
    {
        perror("Error creating save data. Shutting down...");
        exit(EXIT_FAILURE);
    }
    fprintf(f, "%d ", network->size);
    for (int i = 0; i < network->size; i++)
    {
        fprintf(f, "%d ", network->neurons_per_layer[i]);
    }
    fprintf(f, "\n");
    for (int i = 0; i < network->size -1; i++)
    {
        Matrix *w = network->weights[i];
        for (int r = 0; r < w->rows; r++)
            for (int c = 0; c < w->columns; c++)
                fprintf(f, "%f ", w->vectors[c]->elements[r]);
        fprintf(f, "\n");
        Vector *b = network->biases[i];
        for (int j = 0; j < b->size; j++)
            fprintf(f, "%f ", b->elements[j]);
        fprintf(f, "\n");
    }
    fclose(f);
}

void load_network_data(NeuralNetwork *network)
{
    char filename[256];
    sprintf(filename, "data/%c.txt", network->identifier);
    
    FILE *f = fopen(filename, "r");
    if (!f) 
    { 
        perror("Load failed. Shutting down...");
        fclose(f);
        exit(EXIT_FAILURE);
    }

    int size;
    fscanf(f, "%d\n", &size);
    if (size != network->size) {
        printf("Network size mismatch. Shutting down...");
        fclose(f);
        exit(EXIT_FAILURE);
        return;
    }
    for (int s = 0; s < network->size; s++) {
        int expected = network->neurons_per_layer[s];
        int check;
        fscanf(f, "%d\n", &check);
        if (check != expected) {
            printf("Mismatch at layer %d: expected %d neurons, got %d. Shutting down...", s, expected, check);
            fclose(f);
            exit(EXIT_FAILURE);
            return;
        }
    }
    for (int i = 0; i < network->size - 1; i++) {
        Matrix *w = network->weights[i];
        for (int r = 0; r < w->rows; r++) {
            for (int c = 0; c < w->columns; c++) {
                fscanf(f, "%f", &w->vectors[c]->elements[r]);
            }
        }
        Vector *b = network->biases[i];
        for (int j = 0; j < b->size; j++) {
            fscanf(f, "%f", &b->elements[j]);
        }
    }
    fclose(f);
}

int main()
{
    NeuralNetwork *network = init_network((int[]){IMAGE_SIZE,64,32,10},4, 'a');

    Vector *input = create_v(IMAGE_SIZE, (float[]){0}, ZERO);
    float input_array[IMAGE_SIZE];
    int label;
    int offset = 1;
    float learning_rate = 0.05f;

    char command[MAX_COMMAND_LENGTH];
    float input_data[IMAGE_SIZE] = {0};
 
    system("cls");
    printf("Welcome to the Neural CLI Interface\n");

    while (true) 
    {
        printf("\n> ");
        if (fgets(command, MAX_COMMAND_LENGTH, stdin) == NULL) {continue;}
        command[strcspn(command, "\n")] = 0;
        system("cls");
        
        if (strcmp(command, "exit") == 0) 
        {
            printf("Exiting program.\n");
            break;
        } 
        else if (strcmp(command, "load") == 0) 
        {
            load_network_data(network);
            printf("Loading complete!\n");
        }
        else if (strcmp(command, "save") == 0) 
        {
            save_network_data(network);
            printf("Saving complete!\n");
        }
        else if (strcmp(command, "t100") == 0) 
        {
            train_network_batch(network, input, input_array, &label, &offset, 100, learning_rate);
        }
        else if (strcmp(command, "t1000") == 0) 
        {
            train_network_batch(network, input, input_array, &label, &offset, 1000, learning_rate);
        }
        else if (strcmp(command, "t5000") == 0) 
        {
            train_network_batch(network, input, input_array, &label, &offset, 5000, learning_rate);
        }
        else if (strcmp(command, "t10000") == 0) 
        {
            train_network_batch(network, input, input_array, &label, &offset, 10000, learning_rate);
        }
        else if (strcmp(command, "tall") == 0) 
        {
            train_network_batch(network, input, input_array, &label, &offset, 60000, learning_rate);
        }
        else if (strcmp(command, "mat") == 0) 
        {
            print_network_weights(network, false);
        }
        else if (strcmp(command, "vec") == 0) 
        {
            print_network_vectors(network, false);
        }
        else if (strcmp(command, "draw") == 0)
        {
            start_draw_window(network);
        }
        else if (strcmp(command, "i100") == 0) 
        {
            get_input_data(input_array, 0, &label);
            setvalues_v(input, input_array, IMAGE_SIZE);
            set_network_input(network, input);
            set_desired_single_output(network, label);
            
            for (int i = 0; i < 100; i++)
            {
                compute_activation(network);
                backpropagate(network, learning_rate);
        
                system("cls");
                printf("Iteration %d/60000\n", i + offset);
                print_network_output(network);
                Sleep(10);
            }
        }
        else 
        {
            printf("Unknown command: %s\n", command);
        }
    }
    dispose_network(network);
    dispose_v(input);
    return 0;
}