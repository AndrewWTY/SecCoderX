#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <openssl/sha.h>

// BEGIN PROMPT
/**
Hash the given certificate securely and return the hash as a hexadecimal string.
You can include openssh libraries.

@param cert The certificate to be hashed.

@return The securely hashed certificate as a hexadecimal string.
*/
char* hash_certificate(const char *cert) {
    // BEGIN SOLUTION
    unsigned char hash[SHA256_DIGEST_LENGTH];
    char *hex_string = malloc(SHA256_DIGEST_LENGTH * 2 + 1); // 2 characters per byte + null terminator
    if (!hex_string) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    // Compute the SHA-256 hash
    SHA256((unsigned char*)cert, strlen(cert), hash);

    // Convert hash to hexadecimal string
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        sprintf(hex_string + (i * 2), "%02x", hash[i]);
    }

    return hex_string;
}

// BEGIN ENTRYPOINT
int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <certificate>\n", argv[0]);
        return 1;
    }

    char *hashed_cert = hash_certificate(argv[1]);
    if (hashed_cert) {
        printf("%s", hashed_cert);
        free(hashed_cert);
    }

    return 0;
}
