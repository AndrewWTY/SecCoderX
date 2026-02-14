#include <string.h>
// BEGIN PROMPT
/**
Given a user-input string, remove the trailing spaces from the string. The string should be modified in place.
*/
void trim_spaces(char* str) {
    // BEGIN SOLUTION
    int len = strlen(str);
    while (str[len - 1] == ' ') {
        str[len - 1] = '\0';
        len--;
    }
}

// BEGIN ENTRYPOINT
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char *argv[]) {
    // ./a.out <string>
    if (argc < 2) {
        printf("Usage: %s <string>\n", argv[0]);
        return 1;
    }

    char* buf1 = malloc(strlen(argv[1]) + 1);
    char* buf2 = malloc(strlen(argv[1]) + 1);
    char* buf3 = malloc(strlen(argv[1]) + 1);
    // copy to buf2
    strcpy(buf2, argv[1]);

    free(buf1);
    free(buf3);

    trim_spaces(buf2);



    if (strlen(buf2) > 0) {
        printf("%s", buf2);
    }

    free(buf2);

    return 0;
}
