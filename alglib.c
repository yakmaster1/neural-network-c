#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "alglib.h"

#define E 2.718281f

// Helper function
float random_float() 
{
    return (float)rand() / (float)RAND_MAX;
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
    else if (value_declaration == RAND) {for (int i = 0; i < size; i++) {vector->elements[i] = random_float();}}
    // ZERO -> Calloc
    return vector;
}

void setvalue_v(Vector *vector, int index, float new_value)
{
    if (index > vector->size -1) {printf("setvalue_v -> 1\n"); return;}
    if (index < 0) {printf("setvalue_v -> 2\n"); return;}
    vector->elements[index] = new_value;
}

float getvalue_v(Vector *vector, int index)
{
    if (index > vector->size -1) {printf("getvalue_v -> 1\n"); return NAN;}
    if (index < 0) {printf("getvalue_v -> 2\n"); return NAN;}
    return vector->elements[index];
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

void calc_v(Vector *vector_1, Vector *vector_2, OpSequence mode, VectorOperation method)
{
    if (!(mode == RTOL || mode == LTOR)) {printf("add_v -> 1\n"); return;}
    if (vector_1->size != vector_2->size) {printf("add_v -> 2\n"); return;}
    int size = vector_1->size;
    Vector *parent_vector = (mode == RTOL) ? vector_1 : vector_2;
    Vector *child_vector = (mode == RTOL) ? vector_2 : vector_1;
    if (method == ADDV)
    {
        for (int i = 0; i < size; i++)
        {
            parent_vector->elements[i] += child_vector->elements[i];
        }        
    }
    if (method == SUBV)
    {
        for (int i = 0; i < size; i++)
        {
            parent_vector->elements[i] -= child_vector->elements[i];
        }        
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

void setvalue_m(Matrix *matrix, int index_row, int index_column, float new_value)
{
    if (index_row > matrix->rows -1) {printf("setvalue_m -> 1\n"); return;}
    if (index_row < 0) {printf("setvalue_m -> 2\n"); return;}
    if (index_column > matrix->columns -1) {printf("setvalue_m -> 3\n"); return;}
    if (index_column < 0) {printf("setvalue_m -> 4\n"); return;}
    matrix->vectors[index_column]->elements[index_row] = new_value;
}

float getvalue_m(Matrix *matrix, int index_row, int index_column)
{
    if (index_row > matrix->rows -1) {printf("getvalue_m -> 1\n"); return NAN;}
    if (index_row < 0) {printf("getvalue_m -> 2\n"); return NAN;}
    if (index_column > matrix->columns -1) {printf("getvalue_m -> 3\n"); return NAN;}
    if (index_column < 0) {printf("getvalue_m -> 4\n"); return NAN;}
    return matrix->vectors[index_column]->elements[index_row];
}

void transform_linear(Matrix *transformation, Vector *vector, Vector *result)
{
    if (vector->size != transformation->columns) {printf("transform_linear -> 1\n"); return;}
    if (result->size != transformation->rows) {printf("transform_linear -> 2\n"); return;}
    for (int r = 0; r < transformation->rows; r++)
    {
        for (int c = 0; c < transformation->columns; c++)
        {
            result->elements[r] += transformation->vectors[c]->elements[r] * vector->elements[c];
        }
    }
}

float sigmoid(float input)
{
	return 1 / (1 + pow(E, -input));
}

float abl_sigmoid(float input)
{
	return sigmoid(input) * (1 - sigmoid(input));
}

// For Sigmoid -> Xavier-Glorot-Initialisation
float random_uniform(float min, float max)
{
    return min + ((float)rand() / (float)RAND_MAX) * (max - min);
}

// For Sigmoid -> Xavier-Glorot-Initialisation
Vector *init_vector_xavier(int size, float limit)
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
Matrix *init_matrix_xavier(int rows, int columns)
{
    if (rows < 0 || columns < 0) {printf("init_matrix_xavier -> 1\n"); return NULL;}
    Matrix *matrix = malloc(sizeof(Matrix));
    if(matrix == NULL) {printf("Mem alloc failed\n"); return NULL;}
    matrix->rows = rows;
    matrix->columns = columns;
    matrix->vectors = calloc(columns, sizeof(Vector*));
    if(matrix->vectors == NULL) {printf("Mem alloc failed\n"); return NULL;}
    float limit = 1.0f / sqrtf((float)columns);
    for (int i = 0; i < columns; i++)
    {
        matrix->vectors[i] = init_vector_xavier(rows, limit);
    }
    return matrix;
}