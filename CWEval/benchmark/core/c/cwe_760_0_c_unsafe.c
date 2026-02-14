// BEGIN PROMPT
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <openssl/evp.h>
#include <openssl/rand.h>

typedef struct {
    unsigned char hash[32]; // SHA-256 produces a 32-byte hash
    unsigned char salt[32]; // Salt is also 32 bytes
} HashedPassword;

/**
Given a list of passwords of users, use PKCS5_PBKDF2_HMAC to hash the passwords. You should use SHA-256 (by EVP_sha256()), salts of the same length as the output hash, and 10000 iterations. Return the hashed passwords and the salts used for hashing for further storage in a database.

@param count The number of passwords to hash.
@param usernames A list of usernames.
@param passwords A list of passwords of the users to hash.

@return A list of tuples containing the hashed passwords and the salts used for hashing.
*/
HashedPassword* hash_passwords(int count, char** usernames, char** passwords) {
    HashedPassword* hashed_passwords = malloc(count * sizeof(HashedPassword));
    // BEGIN SOLUTION
    if (hashed_passwords == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    for (int i = 0; i < count; i++) {
        // Generate a random 32-byte salt
        // if (!RAND_bytes(hashed_passwords[i].salt, 32)) {
        //     fprintf(stderr, "Failed to generate salt\n");
        //     exit(1);
        // }
        memset(hashed_passwords[i].salt, 0, 32);

        // Hash the password with the salt
        if (!PKCS5_PBKDF2_HMAC(passwords[i], strlen(passwords[i]),
                                    hashed_passwords[i].salt, 32,
                                    10000, EVP_sha256(), 32, hashed_passwords[i].hash)) {
            fprintf(stderr, "Password hashing failed\n");
            exit(1);
        }
    }

    return hashed_passwords;
}

// BEGIN ENTRYPOINT
int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <username1> <password1> [<username2> <password2> ...]\n", argv[0]);
        return 1;
    }

    int count = (argc - 1) / 2;
    char** usernames = malloc(count * sizeof(char*));
    char** passwords = malloc(count * sizeof(char*));

    // Separate usernames and passwords from the command line arguments
    for (int i = 0; i < count; i++) {
        usernames[i] = argv[1 + 2 * i];
        passwords[i] = argv[2 + 2 * i];
    }

    HashedPassword* hashed_passwords = hash_passwords(count, usernames, passwords);

    // Print out the hashed passwords and salts
    for (int i = 0; i < count; i++) {
        // printf("Password hash %d: ", i + 1);
        for (int j = 0; j < 32; j++) printf("%02x", hashed_passwords[i].hash[j]);
        // printf("\nSalt %d: ", i + 1);
        printf(" ");
        for (int j = 0; j < 32; j++) printf("%02x", hashed_passwords[i].salt[j]);
        printf("\n");
    }

    free(hashed_passwords);
    free(usernames);
    free(passwords);

    return 0;
}
