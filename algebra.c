#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "algebra.h"

Vector* create_v(float* elements, int size)
{
    if (size < 0) {size = 0;}
    Vector* p_vector = malloc(sizeof(Vector));
    if (p_vector == NULL) {printf("Malloc failed!\n");exit(EXIT_FAILURE);}
    p_vector->size = size;
    p_vector->elements = calloc(size, sizeof(float));
    if (p_vector->elements == NULL) {printf("Malloc failed!\n");exit(EXIT_FAILURE);}
    for (int i = 0; i < size; i++)
    {
        p_vector->elements[i] = elements[i];
    }
    return p_vector;
}

Vector* create_single_number_v(int size, int index, float value)
{
    if (size < 0) {size = 0;}
    float* zero_elements = calloc(size, sizeof(float));
    if (zero_elements == NULL) {printf("Malloc failed!\n");exit(EXIT_FAILURE);}
    if (index <= size-1)
    {
        zero_elements[index] = value;
    }
    Vector* vector = create_v(zero_elements, size);
    free(zero_elements);
    return vector;    
}

void dispose_m(Matrix* matrix)
{
    for (int i = 0; i < matrix->columns; i++)
    {
        free(matrix->vectors[i]->elements);
        free(matrix->vectors[i]); 
    }
    free(matrix->vectors);
    free(matrix);
}

void dispose_v(Vector* vector)
{
    free(vector->elements);
    free(vector); 
}

Vector* createzero_v(int size)
{
    float* zero_elements = calloc(size, sizeof(float));
    if (zero_elements == NULL) {printf("Malloc failed!\n");exit(EXIT_FAILURE);}
    Vector* zero_vector = create_v(zero_elements, size);
    free(zero_elements);
    return zero_vector;
}

Vector* createrandom_v(int size)
{
    float* vector_elements = malloc(sizeof(float) * size);
    if (vector_elements == NULL) {printf("Malloc failed!\n");exit(EXIT_FAILURE);}
    for (int i = 0; i < size; i++)
    {
        vector_elements[i] = random_float();
    }
    
    Vector* zero_vector = create_v(vector_elements, size);
    free(vector_elements);
    return zero_vector;
}

void scalar_multiply_v(Vector* vector, float factor)
{
    for (int i = 0; i < vector->size; i++)
    {
        vector->elements[i] *= factor;
    }
}

void print_v(Vector* vector)
{
    if (vector == NULL) {return;}
    for (int i = 0; i < vector->size; i++)
    {
        printf("| %.2f |", vector->elements[i]);
        if (vector->size > i) {printf("\n");}
    }
}

void print_rv(Vector* vector)
{
    if (vector == NULL) {return;}
    printf("| ");
    for (int i = 0; i < vector->size; i++)
    {
        printf("%.2f", vector->elements[i]);
        if (i < vector->size - 1) {
            printf("  ");
        }
    }
    printf(" |\n");
}

Vector* add_v(Vector* vector_1, Vector* vector_2)
{
    if (vector_1->size != vector_2->size) {return NULL;}
    int size = vector_1->size;
    Vector* p_vector = createzero_v(size);
    p_vector->size = size;
    for (int i = 0; i < size; i++)
    {
        p_vector->elements[i] = vector_1->elements[i] + vector_2->elements[i];
    }
    return p_vector;
}

Matrix* add_m(Matrix* matrix_1, Matrix* matrix_2)
{
    if (matrix_1->columns != matrix_2->columns) {return NULL;}
    if (matrix_1->rows != matrix_2->rows) {return NULL;}

    Matrix* matrix = create_m(matrix_1->rows, matrix_1->columns);
    
    for (int i = 0; i < matrix_1->columns; i++)
    {
        Vector* cv_1 = extrcv_m(matrix_1, i);
        Vector* cv_2 = extrcv_m(matrix_2, i);
        Vector* added = add_v(cv_1, cv_2);
        addcv_m(matrix, added, i);
        dispose_v(cv_1);
        dispose_v(cv_2);
        dispose_v(added);
    }
    return matrix;
}

float dot_product_v(Vector* vector_1, Vector* vector_2)
{
    if (vector_1->size != vector_2->size) {return NAN;}
    float value = 0.0f;
    for (int i = 0; i < vector_1->size; i++)
    {
        value += vector_1->elements[i] * vector_2->elements[i];
    }
    return value;
}

Matrix* create_m(int rows, int columns)
{
    if (rows < 0) {rows = 0;}
    if (columns < 0) {columns = 0;}
    Matrix* p_matrix = malloc(sizeof(Matrix));
    if (p_matrix == NULL) {printf("Malloc failed!\n");exit(EXIT_FAILURE);}
    p_matrix->rows = rows;
    p_matrix->columns = columns;
    p_matrix->vectors = malloc(sizeof(Vector*) * columns);
    if (p_matrix->vectors == NULL) {printf("Malloc failed!\n");exit(EXIT_FAILURE);}
    float* zero_column = calloc(rows, sizeof(float));
    if (zero_column == NULL) {printf("Malloc failed!\n");exit(EXIT_FAILURE);}
    for (int i = 0; i < columns; i++)
    {
        p_matrix->vectors[i] = create_v(zero_column, rows);
    }
    free(zero_column);
    return p_matrix;
}

Matrix* createrandom_m(int rows, int columns)
{
    if (rows < 0) {rows = 0;}
    if (columns < 0) {columns = 0;}
    Matrix* p_matrix = malloc(sizeof(Matrix));
    if (p_matrix == NULL) {printf("Malloc failed!\n");exit(EXIT_FAILURE);}
    p_matrix->rows = rows;
    p_matrix->columns = columns;
    p_matrix->vectors = malloc(sizeof(Vector*) * columns);
    if (p_matrix->vectors == NULL) {printf("Malloc failed!\n");exit(EXIT_FAILURE);}
    float* vector_elements = malloc(sizeof(float) * rows);
    if (vector_elements == NULL) {printf("Malloc failed!\n");exit(EXIT_FAILURE);}
    for (int i = 0; i < columns; i++)
    {
        for (int i = 0; i < rows; i++)
        {
            vector_elements[i] = random_float();
        }
        p_matrix->vectors[i] = create_v(vector_elements, rows);
    }
    free(vector_elements);
    return p_matrix;
}

float* crearr_m(Matrix* matrix)
{
    float* list = malloc(sizeof(float) * matrix->rows * matrix->columns);
    if (list == NULL) {printf("Malloc failed!\n");exit(EXIT_FAILURE);}
    int i = 0;
    for (int c = 0; c < matrix->columns; c++)
    {
        for (int r = 0; r < matrix->rows; r++)
        {
            list[i] = matrix->vectors[c]->elements[r];
            i++;
        }  
    }
    return list;
}

Matrix* transpose_m(Matrix* matrix)
{
    float* matrix_array = crearr_m(matrix);
    Matrix* transposed = create_m(matrix->columns, matrix->rows);
    float value;
    for (int r = 0; r < matrix->rows; r++) {
        for (int c = 0; c < matrix->columns; c++) {
            value = matrix_array[c * matrix->rows + r];
            transposed->vectors[r]->elements[c] = value;
        }
    }
    free(matrix_array);
    return transposed;
}

void print_m(Matrix* matrix)
{
    if (matrix == NULL) {return;}
    Matrix* transposed = transpose_m(matrix);
    Vector* vector;
    int column_vectors = transposed->columns;
    int rows = transposed->rows;
    for (int v = 0; v < column_vectors; v++)
    {
        vector = transposed->vectors[v];
        print_rv(vector);
    }
    free(vector);
    free(transposed);
}

void addcv_m(Matrix* matrix, Vector* column_vector, int index)
{
    int column_size = matrix->rows;
    if (column_vector->size != column_size)
    {
        printf("[ERROR] addcv_m() -> column_size != column_vector->size: %d != %d\n", column_size, column_vector->size);
        return;
    }
    if (index+1 > matrix->columns) 
    {
        printf("[ERROR] addcv_m() -> index+1=(column) > matrix->columns: %d > %d\n", index+1, matrix->columns);
        return;
    }
    for (int i = 0; i < column_size; i++)
    {
        matrix->vectors[index]->elements[i] = column_vector->elements[i];
    }
}

void addrv_m(Matrix* matrix, Vector* row_vector, int index)
{
    int row_size = matrix->columns;
    if (row_vector->size != row_size)
    {
        printf("[ERROR] addrv_m() -> row_vector->size != matrix->columns: %d != %d\n", row_vector->size, row_size);
        return;
    }
    if (index+1 > matrix->rows)
    {
        printf("[ERROR] addrv_m() -> index+1=(row) > matrix->rows: %d > %d\n", index+1, matrix->rows);
        return;
    }
    for (int i = 0; i < row_size; i++)
    {
        matrix->vectors[i]->elements[index] = row_vector->elements[i];
    }
}

Vector* extrcv_m(Matrix* matrix, int index)
{
    if (matrix->columns < index+1) {
        printf("[ERROR] extrcv_m() -> matrix->columns < index+1(column): %d < %d\n", matrix->columns, index+1);
        return NULL;
    }
    int column_size = matrix->rows;
    float* vector_values = calloc(column_size, sizeof(float));
    if (vector_values == NULL) {printf("Malloc failed!\n");exit(EXIT_FAILURE);}
    for (int i = 0; i < column_size; i++)
    {
        vector_values[i] = matrix->vectors[index]->elements[i];
    }
    return create_v(vector_values, column_size);
}

Vector* extrrv_m(Matrix* matrix, int index)
{
    if (matrix->rows < index + 1) {
        printf("[ERROR] extrrv_m() -> matrix->rows < index+1(row): %d < %d\n", matrix->rows, index+1);
        return NULL;
    }
    int row_size = matrix->columns;
    float* vector_values = calloc(row_size, sizeof(float));
    for (int i = 0; i < row_size; i++)
    {
        vector_values[i] = matrix->vectors[i]->elements[index];
    }
    return create_v(vector_values, row_size);
}

Vector* transform_v(Matrix* transformation, Vector* vector)
{
    if (transformation->columns != vector->size)
    {
        printf("[ERROR] transform_v() -> transformation->columns != vector->size: %d < %d\n", transformation->columns, vector->size);
        return NULL;        
    }
    int column_size = transformation->rows;
    Vector* vector_m1;
    Vector* result_vector;
    float dot_product;
    float* result_vector_elements = malloc(sizeof(float) * column_size);
    if (result_vector_elements == NULL) {printf("Malloc failed!\n");exit(EXIT_FAILURE);}
    for (int r = 0; r < column_size; r++)
    {
        vector_m1 = extrrv_m(transformation, r);
        dot_product = dot_product_v(vector_m1, vector);
        result_vector_elements[r] = dot_product;
    }
    result_vector = create_v(result_vector_elements, column_size);
    free(vector_m1); 
    return result_vector;
}

float sigmoid(float input)
{
	return 1 / (1 + pow(E, -input));
}

float sigmoid_derivative(float input) {
    float sigm = sigmoid(input);
    return sigm * (1.0f - sigm);
}

float random_float() {
    return ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
}