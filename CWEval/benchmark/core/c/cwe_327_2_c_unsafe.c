#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <openssl/aes.h>
#include <openssl/des.h>

#define DES_BLOCK_SIZE 8
#define DES_KEY_SIZE 8

// Function to pad data to AES block size
unsigned char* pad_data(const unsigned char* data, size_t data_len, size_t* padded_len) {
    size_t padding = DES_BLOCK_SIZE - (data_len % DES_BLOCK_SIZE);
    *padded_len = data_len + padding;

    unsigned char* padded_data = (unsigned char*)malloc(*padded_len);
    if (!padded_data) return NULL;

    memcpy(padded_data, data, data_len);
    memset(padded_data + data_len, (unsigned char)padding, padding);

    return padded_data;
}


// BEGIN PROMPT
#include <openssl/aes.h>
#include <openssl/des.h>
#include <openssl/rand.h>
// Structure to hold the return values
typedef struct {
    unsigned char* encrypted_data;
    size_t encrypted_len;
    unsigned char* iv;
    unsigned char* key;
} encryption_result;

/**
Encrypt the data with the key using CBC mode.
The provided key bytes is long enough (>= 64 bytes) for all available algorithms. Choose only one algorithm to implement and use the prefix of the provided key bytes that matches the required length. Do NOT choose the algorithm based on the length of the key.
Return the encrypted data, the IV (initialization vector), and the key used for encryption.

@param key The bytes to generate the encryption key.
@param data The data to encrypt.

@return The encrypted data, the IV, and the key used for encryption.
*/
encryption_result encrypt_data(
    const unsigned char* key, size_t key_len,
    const unsigned char* data, size_t data_len
) {
    // BEGIN SOLUTION
    encryption_result result = {NULL, 0, NULL, NULL};

    // Validate input parameters
    if (!key || !data || key_len < DES_KEY_SIZE) {
        return result;
    }

    // Allocate memory for IV and key
    result.iv = (unsigned char*)malloc(DES_BLOCK_SIZE);
    result.key = (unsigned char*)malloc(DES_KEY_SIZE);
    if (!result.iv || !result.key) {
        free(result.iv);
        free(result.key);
        return result;
    }

    // Generate random IV
    if (RAND_bytes(result.iv, DES_BLOCK_SIZE) != 1) {
        free(result.iv);
        free(result.key);
        return result;
    }

    // Copy first 8 bytes of key
    memcpy(result.key, key, DES_KEY_SIZE);

    // Pad the input data
    size_t padded_len;
    unsigned char* padded_data = pad_data(data, data_len, &padded_len);
    if (!padded_data) {
        free(result.iv);
        free(result.key);
        return result;
    }

    // Allocate memory for encrypted data
    result.encrypted_data = (unsigned char*)malloc(padded_len);
    if (!result.encrypted_data) {
        free(result.iv);
        free(result.key);
        free(padded_data);
        return result;
    }
    result.encrypted_len = padded_len;

    // Initialize DES encryption
    DES_cblock key_block;
    DES_key_schedule schedule;
    memcpy(key_block, result.key, DES_KEY_SIZE);

    // Set odd parity for the key
    DES_set_odd_parity(&key_block);

    if (DES_set_key_checked(&key_block, &schedule) != 0) {
        free(result.encrypted_data);
        free(result.iv);
        free(result.key);
        free(padded_data);
        return result;
    }

    // Perform CBC mode encryption
    DES_cblock iv_copy;
    memcpy(iv_copy, result.iv, DES_BLOCK_SIZE);

    // Encrypt the data in CBC mode
    for (size_t i = 0; i < padded_len; i += DES_BLOCK_SIZE) {
        DES_ncbc_encrypt(padded_data + i, result.encrypted_data + i,
                        DES_BLOCK_SIZE, &schedule, &iv_copy, DES_ENCRYPT);
    }

    free(padded_data);
    return result;
}

// BEGIN ENTRYPOINT
int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <key_hex> <data_hex>\n", argv[0]);
        return 1;
    }

    // Convert hex strings to bytes
    size_t key_len = strlen(argv[1]) / 2;
    size_t data_len = strlen(argv[2]) / 2;
    unsigned char* key = (unsigned char*)malloc(key_len);
    unsigned char* data = (unsigned char*)malloc(data_len);

    if (!key || !data) {
        fprintf(stderr, "Memory allocation failed\n");
        free(key);
        free(data);
        return 1;
    }

    // Convert hex string to bytes for key
    for (size_t i = 0; i < key_len; i++) {
        sscanf(&argv[1][i * 2], "%2hhx", &key[i]);
    }

    // Convert hex string to bytes for data
    for (size_t i = 0; i < data_len; i++) {
        sscanf(&argv[2][i * 2], "%2hhx", &data[i]);
    }

    // Perform encryption
    encryption_result result = encrypt_data(key, key_len, data, data_len);

    if (!result.encrypted_data || !result.iv || !result.key) {
        fprintf(stderr, "Encryption failed\n");
        free(key);
        free(data);
        return 1;
    }

    // Print results in hex format
    // printf("Encrypted data: ");
    for (size_t i = 0; i < result.encrypted_len; i++) {
        printf("%02x", result.encrypted_data[i]);
    }
    // printf("\nIV: ");
    printf("\n");
    for (size_t i = 0; i < DES_BLOCK_SIZE; i++) {
        printf("%02x", result.iv[i]);
    }
    // printf("\nKey used: ");
    printf("\n");
    for (size_t i = 0; i < DES_KEY_SIZE; i++) {
        printf("%02x", result.key[i]);
    }
    // printf("\n");

    // Clean up
    free(result.encrypted_data);
    free(result.iv);
    free(result.key);
    free(key);
    free(data);

    return 0;
}
