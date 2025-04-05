#include "algebra.h"

Vector* create_v(float* elements, int size)
{
    if (size < 0) {size = 0;}
    Vector* p_vector = malloc(sizeof(Vector));
    CHECK_MALLOC(p_vector);
    p_vector->size = size;
    p_vector->elements = calloc(size, sizeof(float));
    CHECK_MALLOC(p_vector->elements);
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
    CHECK_MALLOC(zero_elements);
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

Stcvec stcvec(float* elements, int size)
{
    if (size < 0) {size = 0;}
    Stcvec vector;
    vector.size = size;
    vector.elements = calloc(size, sizeof(float));
    CHECK_MALLOC(vector.elements);
    for (int i = 0; i < size; i++)
    {
        vector.elements[i] = elements[i];
    }
    return vector;    
}

Stcvec stcvec_v(Vector* vector)
{
    return stcvec(vector->elements, vector->size);
}

Vector* createzero_v(int size)
{
    float* zero_elements = calloc(size, sizeof(float));
    CHECK_MALLOC(zero_elements);
    Vector* zero_vector = create_v(zero_elements, size);
    free(zero_elements);
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
    CHECK_MALLOC(p_matrix);
    p_matrix->rows = rows;
    p_matrix->columns = columns;
    p_matrix->vectors = malloc(sizeof(Vector*) * columns);
    CHECK_MALLOC(p_matrix->vectors);
    float* zero_column = calloc(rows, sizeof(float));
    CHECK_MALLOC(zero_column);
    for (int i = 0; i < columns; i++)
    {
        p_matrix->vectors[i] = create_v(zero_column, rows);
    }
    free(zero_column);
    return p_matrix;
}

float* crearr_m(Matrix* matrix)
{
    float* list = malloc(sizeof(float) * matrix->rows * matrix->columns);
    CHECK_MALLOC(list);
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

void addcv_m(Matrix* matrix, Stcvec column_vector, int index)
{
    int column_size = matrix->rows;
    if (column_vector.size != column_size)
    {
        printf("[ERROR] addcv_m() -> column_size != column_vector.size: %d != %d\n", column_size, column_vector.size);
        return;
    }
    if (index+1 > matrix->columns) 
    {
        printf("[ERROR] addcv_m() -> index+1=(column) > matrix->columns: %d > %d\n", index+1, matrix->columns);
        return;
    }
    for (int i = 0; i < column_size; i++)
    {
        matrix->vectors[index]->elements[i] = column_vector.elements[i];
    }
    free(column_vector.elements);
}

void addrv_m(Matrix* matrix, Stcvec row_vector, int index)
{
    int row_size = matrix->columns;
    if (row_vector.size != row_size)
    {
        printf("[ERROR] addrv_m() -> row_vector.size != matrix->columns: %d != %d\n", row_vector.size, row_size);
        return;
    }
    if (index+1 > matrix->rows)
    {
        printf("[ERROR] addrv_m() -> index+1=(row) > matrix->rows: %d > %d\n", index+1, matrix->rows);
        return;
    }
    for (int i = 0; i < row_size; i++)
    {
        matrix->vectors[i]->elements[index] = row_vector.elements[i];
    }
    free(row_vector.elements);
}

Vector* extrcv_m(Matrix* matrix, int index)
{
    if (matrix->columns < index+1) {
        printf("[ERROR] extrcv_m() -> matrix->columns < index+1(column): %d < %d\n", matrix->columns, index+1);
        return NULL;
    }
    int column_size = matrix->rows;
    float* vector_values = calloc(column_size, sizeof(float));
    CHECK_MALLOC(vector_values);
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

Matrix* mult_m(Matrix* m1, Matrix* m2)
{
    if (m1->columns != m2->rows)
    {
        printf("[ERROR] mult_m() -> m1->columns != m2->rows: %d < %d\n", m1->columns, m2->rows);
        return NULL;        
    }
    int column_size = m1->rows;
    int row_size = m2->columns;
    Matrix* result = create_m(column_size, row_size);
    Vector* vector_m1;
    Vector* vector_m2;
    Stcvec column_vector;
    float dot_product;
    float* column_vector_elements = malloc(sizeof(float) * column_size);
    CHECK_MALLOC(column_vector_elements);
    for (int c = 0; c < row_size; c++)
    {
        for (int r = 0; r < column_size; r++)
        {
            vector_m1 = extrrv_m(m1, r);
            vector_m2 = extrcv_m(m2, c);
            dot_product = dot_product_v(vector_m1, vector_m2);
            column_vector_elements[r] = dot_product;
        }
        column_vector = stcvec(column_vector_elements, column_size);
        addcv_m(result, column_vector, c);
    }
    free(vector_m1);
    free(vector_m2); 
    return result;
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
    CHECK_MALLOC(result_vector_elements);
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

void sigmoid_v(Vector* vector)
{
    for (int i = 0; i < vector->size; i++)
    {
        vector->elements[i] = sigmoid(vector->elements[i]);
    }
}