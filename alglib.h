#ifndef ALGLIB_H
#define ALGLIB_H

#include <stdbool.h>

typedef struct Vector Vector;
typedef struct Matrix Matrix;

typedef enum OpSequence OpSequence;
typedef enum VectorOperation VectorOperation;
typedef enum VectorDeclaration VectorDeclaration;

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

enum VectorOperation 
{
    ADDV = 0,
    SUBV = 1,
    HPROD = 2
};

enum VectorDeclaration 
{
    ZERO = 0,
    RAND = 1,
    INIT = 2
};

Vector *create_v(int size, float *elements, VectorDeclaration value_declaration);
Matrix *create_m(int rows, int columns, VectorDeclaration value_declaration);
void dispose_v(Vector *vector);
void dispose_m(Matrix *matrix);
void print_v(Vector *vector);
void printmultiple_v(int amount, Vector **vectors, int space, bool use_divider);
void print_m(Matrix* matrix);

void transform_linear(Matrix *transformation, Vector *vector, Vector *result);

float sigmoid(float input);
float abl_sigmoid(float input);
Matrix *init_matrix_xavier(int outputs, int inputs);

#endif