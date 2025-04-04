#ifndef WIN_EXT
#define WIN_EXT

#include <windows.h>

#define CHECK_MALLOC(ptr) \
    if ((ptr) == NULL) { \
        fprintf(stderr, "malloc failed at %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

void clear_console();

#endif