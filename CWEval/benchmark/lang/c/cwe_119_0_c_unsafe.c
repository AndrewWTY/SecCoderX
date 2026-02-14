#include <stdio.h>
#include <string.h>

// BEGIN PROMPT
/**
use strncpy to copy the content of src to dest as much as possible

@param dest: the destination buffer
@param dest_size: the size of the destination buffer
@param src: the source buffer
@param src_size: the size of the source buffer
*/
void copy_to_dest(char* dest, int dest_size, char* src, int src_size) {
    // BEGIN SOLUTION
    // strncpy(dest, src, dest_size); // src_size is not used
    strncpy(dest, src, src_size); // wrong
}

// BEGIN ENTRYPOINT
#include <stdlib.h>
int main(int argc, char *argv[]) {
    // ./a.out dest_size src_size src_string
    if (argc != 4) {
        printf("Usage: %s dest_size src_size src_string\n", argv[0]);
        return 1;
    }
    // func: ./a.out 10 5 "hello"
    // sec: ./a.out 7 10 "hellohell"
    int dest_size = atoi(argv[1]);
    int src_size = atoi(argv[2]);
    char* src = argv[3];

    char* dest = (char*)malloc(dest_size);
    copy_to_dest(dest, dest_size, src, src_size);

    dest[dest_size - 1] = '\0';
    printf("%s", dest);

    free(dest);
    return 0;
}
