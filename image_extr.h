#ifndef IMAGE_EXTR_H
#define IMAGE_EXTR_H

#define IMAGE_ROWS 28
#define IMAGE_COLS 28
#define IMAGE_SIZE (IMAGE_ROWS * IMAGE_COLS)

void get_input_data(float *input_data, int index, int *label_out);

#endif