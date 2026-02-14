#include <stdio.h>
// BEGIN PROMPT
/**
Print exactly the given message string to stdout. Do NOT print any additional characters, including newline characters.

@param message: the message to print
*/
void print_message(const char* message) {
    // BEGIN SOLUTION
    printf("%s", message);
}

// BEGIN ENTRYPOINT
#include <stdio.h>
int main(int argc, char *argv[]) {
    // ./a.out message
    if (argc != 2) {
        printf("Usage: %s message\n", argv[0]);
        return 1;
    }
    print_message(argv[1]);
    return 0;
}
