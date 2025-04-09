#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "alglib.h"

#define E 2.718281f

// For Sigmoid -> Xavier-Glorot-Initialisation
float random_uniform(float min, float max)
{
    return min + ((float)rand() / (float)RAND_MAX) * (max - min);
}

// Helper function
void print_offset(int biggest_offset, int element_offset)
{
    for (int i = 0; i < biggest_offset-element_offset; i++) {printf(" ");}
}

// Helper function
int get_size_of_float(float val)
{
    if (val == 0.0f) {return signbit(val) ? 2 : 1;}
    int digits = (int)log10f(fabsf(val)) + 1;
    if (digits < 1) digits = 1;
    if (val < 0.0f) {digits += 1;}
    return digits;
}

Vector *create_v(int size, float *elements, VectorDeclaration value_declaration)
{
    if (size < 1) {printf("create_v -> 2\n"); return NULL;}
    Vector *vector = malloc(sizeof(Vector));
    if (vector == NULL) {printf("Mem alloc failed\n"); return NULL;}
    vector->size = size;
    vector->elements = calloc(size, sizeof(float));
    if (vector->elements == NULL) {printf("Mem alloc failed\n"); return NULL;}
    if (value_declaration == INIT) {for (int i = 0; i < size; i++) {vector->elements[i] = elements[i];}}
    else if (value_declaration == RAND) {for (int i = 0; i < size; i++) {vector->elements[i] = random_uniform(0,1);}}
    // ZERO -> Calloc
    return vector;
}

void dispose_v(Vector *vector)
{
    if (vector == NULL) {printf("dispose_v -> 1\n"); return;}
    free(vector->elements);
    free(vector);
}

void print_v(Vector *vector)
{
    int biggest_offset = 0;
    int total_offset = 0;
    int *element_offsets = calloc(vector->size, sizeof(int));
    if (element_offsets == NULL) {printf("Mem alloc failed\n"); return;}
    for (int i = 0; i < vector->size; i++) {
        float val = vector->elements[i];
        int element_offset = 1;
        if (fabsf(val) > 1e-6f) {
            int digits_before_dot = (val == 0.0f) ? 1 : (int)log10f(fabsf(val)) + 1;
            total_offset = digits_before_dot + 1 + 2 + (val < 0.0f ? 1 : 0);
        }
        if (vector->elements[i] < 0.0f) {element_offset++;}
        if (element_offset > biggest_offset) {biggest_offset = element_offset;}
        element_offsets[i] = element_offset;
    }
    for (int i = 0; i < vector->size; i++)
    {
        printf("|");
        print_offset(biggest_offset, total_offset);
        printf(" %.2f ", vector->elements[i]);
        printf("|");
        if (i < vector->size -1) {printf("\n");}
    }
    free(element_offsets);
}

void printmultiple_v(int amount, Vector **vectors, int space, bool use_divider)
{
    if (amount < 1) { printf("printmultiple_v -> 1\n"); return; }
    if (space < 0) { printf("printmultiple_v -> 2\n"); return; }
    int size = vectors[0]->size;
    for (int i = 0; i < amount; i++) 
    {
        if (vectors[i]->size != size) {printf("printmultiple_v -> 3\n"); return;}
    }
    int field_width = 7;
    int precision = 2;
    int added_offset = 0;
    for (int a = 0; a < amount; a++)
    {
        for (int s = 0; s < vectors[a]->size; s++)
        {
            int float_size = get_size_of_float(vectors[a]->elements[s]);
            if (float_size >= added_offset) {added_offset = float_size;}
        }
    }
    for (int s = 0; s < size; s++) 
    {
        printf("|");
        for (int a = 0; a < amount; a++) 
        {
            if (a != 0) 
            {
                if (use_divider) printf("|");
                for (int i = 0; i < space; i++) printf(" ");
                if (use_divider) printf("|");
            }
            printf(" %*.*f ", field_width + added_offset - 4, precision, vectors[a]->elements[s]);
        }
        printf("|");
        if (s < size - 1) printf("\n");
    }
}

Matrix *create_m(int rows, int columns, VectorDeclaration value_declaration)
{
    if (rows < 0 || columns < 0) {printf("create_m -> 1\n"); return NULL;}
    Matrix *matrix = malloc(sizeof(Matrix));
    if(matrix == NULL) {printf("Mem alloc failed\n"); return NULL;}
    matrix->rows = rows;
    matrix->columns = columns;
    matrix->vectors = calloc(columns, sizeof(Vector*));
    if(matrix->vectors == NULL) {printf("Mem alloc failed\n"); return NULL;}
    VectorDeclaration declaration = (value_declaration == 2) ? 0 : value_declaration;
    for (int i = 0; i < columns; i++)
    {
        matrix->vectors[i] = create_v(rows, (float[]){0}, declaration);
    }
    return matrix;    
}

void print_m(Matrix* matrix)
{
    printmultiple_v(matrix->columns, matrix->vectors, 1, false);
}

void dispose_m(Matrix *matrix)
{
    if(matrix == NULL) {printf("dispose_m -> 1\n"); return;}
    for (int i = 0; i < matrix->columns; i++)
    {
        dispose_v(matrix->vectors[i]);
    }
    free(matrix->vectors);
    free(matrix);    
}

void transform_linear(Matrix *transformation, Vector *vector, Vector *result)
{
    if (vector->size != transformation->columns) {printf("transform_linear -> 1\n"); return;}
    if (result->size != transformation->rows) {printf("transform_linear -> 2\n"); return;}
    for (int r = 0; r < transformation->rows; r++)
    {
        result->elements[r] = 0.0f;
        for (int c = 0; c < transformation->columns; c++)
        {
            result->elements[r] += transformation->vectors[c]->elements[r] * vector->elements[c];
        }
    }
}

void setvalues_v(Vector *vector, float *data, int size)
{
    if (size != vector->size) {printf("setvalues_v -> 1\n"); return;}
    for (int i = 0; i < size; i++)
    {
        vector->elements[i] = data[i];
    }
}

float sigmoid(float input)
{
	return 1.0f / (1.0f + pow(E, -input));
}

float abl_sigmoid(float input)
{
	return sigmoid(input) * (1.0f - sigmoid(input));
}

float relu(float input) 
{
    return input > 0.0f ? input : 0.0f;
}

float abl_relu(float input) 
{
    return input > 0.0f ? 1.0f : 0.0f;
}

// Sigmoid -> Xavier-Glorot-Initialisation
// float limit = 1.0f / sqrtf((float)columns);
// float random_value = random_uniform(-limit, limit);
// Re
Vector *pre_init_vector(int size, float limit)
{
    if (size < 1) {printf("init_vector_xavier -> 2\n"); return NULL;}
    Vector *vector = malloc(sizeof(Vector));
    if(vector == NULL) {printf("Mem alloc failed\n"); return NULL;}
    vector->size = size;
    vector->elements = calloc(size, sizeof(float));
    if(vector->elements == NULL) {printf("Mem alloc failed\n"); return NULL;}
    for (int i = 0; i < size; i++) {
        float random_value = random_uniform(-limit, limit);
        vector->elements[i] = random_value;
    }
    return vector;
}

// For Sigmoid -> Xavier-Glorot-Initialisation
Matrix *pre_init_matrix(int rows, int columns)
{
    if (rows < 0 || columns < 0) {printf("init_matrix_xavier -> 1\n"); return NULL;}
    Matrix *matrix = malloc(sizeof(Matrix));
    if(matrix == NULL) {printf("Mem alloc failed\n"); return NULL;}
    matrix->rows = rows;
    matrix->columns = columns;
    matrix->vectors = calloc(columns, sizeof(Vector*));
    if(matrix->vectors == NULL) {printf("Mem alloc failed\n"); return NULL;}
    //float limit = 1.0f / sqrtf((float)columns);
    float limit = sqrtf(2.0f / columns); // HE init
    for (int i = 0; i < columns; i++)
    {
        matrix->vectors[i] = pre_init_vector(rows, limit);
    }
    return matrix;
}