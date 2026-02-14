#include <string.h>

// BEGIN PROMPT
/**
Given a string, return a duplicate of the string. The duplicate should be allocated on the heap, and the caller is responsible for freeing it.

@param str: the string to duplicate
@return a duplicate of the string
*/
char* my_strdup(const char* str) {
    // BEGIN SOLUTION
    if (str == NULL) {
        return NULL;
    }
    char *copy;
    // allocate memory for the copy and copy the string into it
    copy = (char *)malloc(strlen(str) + 1);
    if (copy == NULL) {
        return NULL;
    }
    strcpy(copy, str);
    return copy;
}

// BEGIN ENTRYPOINT
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char *argv[]) {
    // ./a.out <string>
    if (argc != 2) {
        printf("Usage: %s <string>\n", argv[0]);
        return 1;
    }
    char *str = argv[1];
    char *copy = my_strdup(str);
    if (copy == NULL) {
        return 0;
    }
    printf("%s", copy);
    free(copy);
    return 0;
}
