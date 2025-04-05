#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#include <stdio.h>

#define WIDTH 28
#define HEIGHT 28
#define SIZE (WIDTH * HEIGHT)

unsigned char *get_image(FILE *f, int index);
void test_image(int offset); 

#endif