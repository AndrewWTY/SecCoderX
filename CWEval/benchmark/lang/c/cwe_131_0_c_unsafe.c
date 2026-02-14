#include <string.h>

/**

*/
char *my_strdup(const char *str) {
    if (str == NULL) {
        return NULL;
    }
    char *copy;
    // allocate memory for the copy and copy the string into it
    copy = (char *)malloc(strlen(str));
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
    printf("%s", copy);
    free(copy);
    return 0;
}
