#include "image_loader.h"

#define IMGURL "data/train-images.idx3-ubyte"

unsigned char* get_image(FILE *f, int index) 
{
    static unsigned char image[SIZE];
    unsigned long long offset = 16 + index * SIZE;
    fseek(f, offset, 0);
    fread(image, sizeof(unsigned char), SIZE, f);
    return image;
}

void test_image(int offset) 
{ 
    FILE *f = fopen(IMGURL, "rb");
    if (f == NULL) 
    {
        printf("Datei konnte nicht geöffnet werden.\n");
        return;
    }
    unsigned char *img = get_image(f, 0);
    int pixel;
    for (int row = 0; row < HEIGHT; row++) 
    {
        for (int col = 0; col < WIDTH; col++) 
        {
            pixel = img[row * WIDTH + col];
            printf("%3d ", pixel);
        }
        printf("\n");
    }
    fclose(f);
}

int main() 
{
    test_image(0);
    return 0;
}