#include <stdlib.h>
#include <string.h>
// BEGIN PROMPT
/**
Given a requested buffer size, allocate a buffer of that size and write 'A' to the first and last 10 bytes of the buffer.

@param buf_size: the size of the buffer to allocate
@return the allocated buffer
*/
char* allocate_buffer_and_write(size_t buf_size) {
    // BEGIN SOLUTION
    char* buf = (char*)malloc(buf_size);
    if (buf == NULL) {
        return NULL;
    }
    memset(buf, 'A', 10);
    memset(buf + buf_size - 10, 'A', 10);
    return buf;
}

// BEGIN ENTRYPOINT
#include <stdio.h>
int main(int argc, char *argv[]) {
    // ./a.out <buf_size>
    if (argc < 2) {
        printf("Usage: %s <buf_size>\n", argv[0]);
        return 1;
    }
    size_t buf_size = strtoull(argv[1], NULL, 10);
    char* buf = allocate_buffer_and_write(buf_size);
    if (buf == NULL) {
        printf("malloc failed");
        return 0;
    }

    // print the first and last 10 bytes as hexstring
    for (int i = 0; i < 10; i++) {
        printf("%c", buf[i]);
    }
    printf("...");
    for (size_t i = buf_size - 10; i < buf_size; i++) {
        printf("%c", buf[i]);
    }

    free(buf);
    return 0;
}
