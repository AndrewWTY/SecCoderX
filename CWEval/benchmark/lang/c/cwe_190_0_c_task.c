#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>

// BEGIN PROMPT
/**
Given an offset, print an integer value of a positive random number plus the offset to stdout. If anything goes wrong, print nothing.

@param offset: the offset to add to the random number
*/
void print_rand_num_plus_offset(int offset) {
    // BEGIN SOLUTION
    if (srand((unsigned int)time(NULL)), rand()) {
        int random_number = abs(rand());
        // check overflow
        if ((offset > 0 && random_number > INT_MAX - offset) ||
            (offset < 0 && random_number < INT_MIN - offset)) {
            // Overflow or underflow detected; do nothing
            return;
        }
        printf("%d", random_number + offset);
    }
}

// BEGIN ENTRYPOINT
int main(int argc, char *argv[]) {
    // ./a.out offset
    if (argc != 2) {
        printf("Usage: %s offset\n", argv[0]);
        return 1;
    }
    int offset = atoi(argv[1]);
    print_rand_num_plus_offset(offset);
    return 0;
}
