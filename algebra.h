#ifndef ALGEBRA_H
#define ALGEBRA_H

#define E 2.718281f

typedef struct Vector Vector;
typedef struct Stcvec Stcvec;
typedef struct Matrix Matrix;

struct Vector 
{
    int size;
    float* elements;
};

struct Stcvec
{
    int size;
    float* elements;
};

struct Matrix
{
    int rows;
    int columns;
    Vector** vectors;
};

Vector* create_v(float* elements, int size);
Matrix* create_m(int rows, int columns);
Matrix* createrandom_m(int rows, int columns);
Vector* createzero_v(int size);
Vector* createrandom_v(int size);
void dispose_v(Vector* vector);
void dispose_m(Matrix* matrix);
float sigmoid(float input);
float sigmoid_derivative(float input);
Matrix* transpose_m(Matrix* matrix);
void print_v(Vector* vector);
void print_m(Matrix* matrix);
Vector* transform_v(Matrix* transformation, Vector* vector);
Vector* add_v(Vector* vector_1, Vector* vector_2);
Matrix* add_m(Matrix* matrix_1, Matrix* matrix_2);
Vector* create_single_number_v(int size, int index, float value);
float random_float();

#endif