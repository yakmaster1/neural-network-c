#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct Vector Vector;
typedef struct Matrix Matrix;

typedef enum OpSequence OpSequence;
typedef enum VectorOperation VectorOperation;
typedef enum Bool Bool;

struct Vector 
{
    int size;
    float *elements;
};

struct Matrix
{
    int rows;
    int columns;

    Vector **vectors;
};

enum OpSequence {
    RTOL = 0,
    LTOR = 1
};

enum VectorOperation {
    ADDV = 0,
    SUBV = 1
};

enum Bool {
    FALSE = 0,
    TRUE = 1
};

Vector *create_v(int size, float *elements, Bool is_zero)
{
    if (size < 1) {printf("create_v -> 2\n"); return NULL;}
    Vector *vector = malloc(sizeof(Vector));
    if(vector == NULL) {printf("Mem alloc failed\n"); return NULL;}
    vector->size = size;
    vector->elements = calloc(size, sizeof(float));
    if(vector->elements == NULL) {printf("Mem alloc failed\n"); return NULL;}
    if (is_zero == FALSE) {for (int i = 0; i < size; i++) {vector->elements[i] = elements[i];}}
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
    if(vector == NULL) {printf("dispose_v -> 1\n"); return;}
    free(vector->elements);
    free(vector);
}

// Helper function
void print_offset(int biggest_offset, int element_offset)
{
    for (int i = 0; i < biggest_offset-element_offset; i++) {printf(" ");}
}

void print_v(Vector *vector)
{
    int biggest_offset = 0;
    int *element_offsets = calloc(vector->size, sizeof(int));
    if(element_offsets == NULL) {printf("Mem alloc failed\n"); return;}
    for (int i = 0; i < vector->size; i++) {
        int element_offset = (vector->elements[i] == 0) ? 1 : (int)(log10(abs(vector->elements[i])) + 1);
        if (vector->elements[i] < 0.0f) {element_offset++;}
        if (element_offset > biggest_offset) {biggest_offset = element_offset;}
        element_offsets[i] = element_offset;
    }
    for (int i = 0; i < vector->size; i++)
    {
        printf("|");
        print_offset(biggest_offset, element_offsets[i]);
        printf(" %.2f ", vector->elements[i]);
        printf("|\n");
    }
    free(element_offsets);
}

void printmultiple_v(int amount, Vector **vectors, int space, Bool use_divider)
{
    if (amount < 1) {printf("printmultiple_v -> 1\n"); return;}
    if (space < 0) {printf("printmultiple_v -> 2\n"); return;}
    int size = vectors[0]->size;
    for (int i = 0; i < amount; i++)
    {
        if (vectors[i]->size != size) {printf("printmultiple_v -> 3\n"); return;}
    }
    
    int *biggest_offset = calloc(amount, sizeof(int));
    if(biggest_offset == NULL) {printf("Mem alloc failed\n"); return;}
    int *element_offsets = calloc(size * amount, sizeof(int));
    if(element_offsets == NULL) {printf("Mem alloc failed\n"); return;}
    for (int a = 0; a < amount; a++)
    {
        for (int s = 0; s < size; s++) {
            int element_offset = (vectors[a]->elements[s] == 0) ? 1 : (int)(log10(abs(vectors[a]->elements[s])) + 1);
            if (vectors[a]->elements[s] < 0.0f) {element_offset++;}
            if (element_offset > biggest_offset[a]) {biggest_offset[a] = element_offset;}
            element_offsets[a+(s*amount)] = element_offset;
        }
    }
    for (int s = 0; s < size; s++)
    {
        printf("|");
        for (int a = 0; a < amount; a++)
        {
            if (a != 0) 
            {
                if (use_divider == TRUE) {printf("|");}
                for (int i = 0; i < space; i++)
                {
                    printf(" ");
                }
                if (use_divider == TRUE) {printf("|");}
            }
            print_offset(biggest_offset[a], element_offsets[a+(s*amount)]);
            printf(" %.2f ", vectors[a]->elements[s]);
        }
        printf("|\n");
    }
    free(biggest_offset);
    free(element_offsets);
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

Matrix *create_m(int rows, int columns)
{
    if (rows < 0 || columns < 0) {printf("create_m -> 1\n"); return NULL;}
    Matrix *matrix = malloc(sizeof(Matrix));
    if(matrix == NULL) {printf("Mem alloc failed\n"); return NULL;}
    matrix->rows = rows;
    matrix->columns = columns;
    matrix->vectors = calloc(columns, sizeof(Vector*));
    if(matrix->vectors == NULL) {printf("Mem alloc failed\n"); return NULL;}
    for (int i = 0; i < columns; i++)
    {
        matrix->vectors[i] = create_v(rows, (float[]){0}, TRUE);
    }
    
    return matrix;    
}

void print_m(Matrix* matrix)
{
    printmultiple_v(matrix->columns, matrix->vectors, 1, FALSE);
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

int main()
{
    Matrix *m = create_m(5,3);
    setvalue_m(m,0,2,-55);
    setvalue_m(m,1,1,555);
    print_m(m);

    dispose_m(m);
    m = NULL;
    return 0;
}
