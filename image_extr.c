#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "alglib.h"
#include "image_extr.h"

#define IMAGE_PATH "data/train-images.idx3-ubyte"
#define LABEL_PATH "data/train-labels.idx1-ubyte"

void get_input_data(float *input_data, int index, int *label_out)
{
    FILE *image_file = fopen(IMAGE_PATH, "rb");
    if (!image_file) {
        perror("ERROR opening image file");
        printf("Arbeitsverzeichnis: ");
        system("cd");
        exit(EXIT_FAILURE);
    }
    fseek(image_file, 16 + index * IMAGE_SIZE, SEEK_SET);
    unsigned char buffer[IMAGE_SIZE];
    fread(buffer, sizeof(unsigned char), IMAGE_SIZE, image_file);
    fclose(image_file);

    for (int i = 0; i < IMAGE_SIZE; i++) {
        input_data[i] = (float)buffer[i];
    }
    FILE *label_file = fopen(LABEL_PATH, "rb");
    if (!label_file) {
        perror("ERROR opening label file");
        printf("Arbeitsverzeichnis: ");
        system("cd");
        exit(EXIT_FAILURE);
    }
    fseek(label_file, 8 + index, SEEK_SET);
    unsigned char label_byte;
    fread(&label_byte, sizeof(unsigned char), 1, label_file);
    fclose(label_file);
    *label_out = (int)label_byte;
}

/* int main()
{
    Vector *input = create_v(IMAGE_SIZE, (float[]){0}, ZERO);
    float input_array[IMAGE_SIZE];
    int label;

    get_input_data(input_array, 2, &label);
    setvalues_v(input, input_array, IMAGE_SIZE);

    printf("%d", label);

    dispose_v(input);
} */