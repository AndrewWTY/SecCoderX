#include <iostream>
#include <vector>
#include <iomanip>
#include <openssl/aes.h>
#include <openssl/des.h>
#include <openssl/rand.h>
#include <cstring>

#define DES_BLOCK_SIZE 8
#define DES_KEY_SIZE 8

// Function to pad data to AES block size
std::vector<unsigned char> pad_data(const std::vector<unsigned char>& data, size_t& padded_len) {
    size_t padding = DES_BLOCK_SIZE - (data.size() % DES_BLOCK_SIZE);
    padded_len = data.size() + padding;

    std::vector<unsigned char> padded_data(padded_len);
    std::copy(data.begin(), data.end(), padded_data.begin());
    std::fill(padded_data.begin() + data.size(), padded_data.end(), static_cast<unsigned char>(padding));

    return padded_data;
}

// Structure to hold the return values
struct encryption_result {
    std::vector<unsigned char> encrypted_data;
    size_t encrypted_len;
    std::vector<unsigned char> iv;
    std::vector<unsigned char> key;
};

/**
Encrypt the data with the key using CBC mode.
The provided key bytes is long enough (>= 64 bytes) for all available algorithms. Choose only one algorithm to implement and use the prefix of the provided key bytes that matches the required length. Do NOT choose the algorithm based on the length of the key.
Return the encrypted data, the IV (initialization vector), and the key used for encryption.

@param key The bytes to generate the encryption key.
@param data The data to encrypt.

@return The encrypted data, the IV, and the key used for encryption.
*/
encryption_result encrypt_data(
    const std::vector<unsigned char>& key, size_t key_len,
    const std::vector<unsigned char>& data, size_t data_len
) {
    // BEGIN SOLUTION
    encryption_result result = { {}, 0, {}, {} };

    // Validate input parameters
    if (key.empty() || data.empty() || key_len < DES_KEY_SIZE) {
        return result;
    }

    // Allocate memory for IV and key
    result.iv.resize(DES_BLOCK_SIZE);
    result.key.resize(DES_KEY_SIZE);

    // Generate random IV
    if (RAND_bytes(result.iv.data(), DES_BLOCK_SIZE) != 1) {
        return result;
    }

    // Copy first 8 bytes of key
    std::copy(key.begin(), key.begin() + DES_KEY_SIZE, result.key.begin());

    // Pad the input data
    size_t padded_len;
    std::vector<unsigned char> padded_data = pad_data(data, padded_len);
    if (padded_data.empty()) {
        return result;
    }

    // Allocate memory for encrypted data
    result.encrypted_data.resize(padded_len);
    result.encrypted_len = padded_len;

    // Initialize DES encryption
    DES_cblock key_block;
    DES_key_schedule schedule;
    std::copy(result.key.begin(), result.key.end(), key_block);

    // Set odd parity for the key
    DES_set_odd_parity(&key_block);

    if (DES_set_key_checked(&key_block, &schedule) != 0) {
        return result;
    }

    // Perform CBC mode encryption
    DES_cblock iv_copy;
    std::copy(result.iv.begin(), result.iv.end(), iv_copy);

    // Encrypt the data in CBC mode
    for (size_t i = 0; i < padded_len; i += DES_BLOCK_SIZE) {
        DES_ncbc_encrypt(padded_data.data() + i, result.encrypted_data.data() + i,
                         DES_BLOCK_SIZE, &schedule, &iv_copy, DES_ENCRYPT);
    }

    return result;
}

// BEGIN ENTRYPOINT
int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <key_hex> <data_hex>" << std::endl;
        return 1;
    }

    // Convert hex strings to bytes
    size_t key_len = strlen(argv[1]) / 2;
    size_t data_len = strlen(argv[2]) / 2;
    std::vector<unsigned char> key(key_len);
    std::vector<unsigned char> data(data_len);

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

    if (result.encrypted_data.empty() || result.iv.empty() || result.key.empty()) {
        std::cerr << "Encryption failed" << std::endl;
        return 1;
    }

    // Print results in hex format
    for (size_t i = 0; i < result.encrypted_len; i++) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)result.encrypted_data[i];
    }

    std::cout << std::endl;
    for (size_t i = 0; i < DES_BLOCK_SIZE; i++) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)result.iv[i];
    }

    std::cout << std::endl;
    for (size_t i = 0; i < DES_KEY_SIZE; i++) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)result.key[i];
    }

    std::cout << std::endl;

    return 0;
}
