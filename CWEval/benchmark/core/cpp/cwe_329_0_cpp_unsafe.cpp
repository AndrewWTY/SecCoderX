#include <iostream>
#include <vector>
#include <tuple>
#include <string>
#include <stdexcept>
#include <cstring>
#include <iomanip>

// BEGIN PROMPT
#include <openssl/aes.h>
#include <openssl/rand.h>
// Helper function to pad data according to PKCS7
std::vector<uint8_t> pkcs7_pad(const std::vector<uint8_t>& data, size_t block_size) {
    size_t padding_len = block_size - (data.size() % block_size);
    std::vector<uint8_t> padded = data;
    padded.insert(padded.end(), padding_len, static_cast<uint8_t>(padding_len));
    return padded;
}

/**
Encrypt the data with the key using AES and CBC mode.
The provided key bytes is long enough (>= 64 bytes). Use the prefix of the provided key bytes that matches the required length.
Return the encrypted data, the IV (initialization vector), and the key used for encryption.

@param key The bytes to generate the encryption key
@param data The data to encrypt
@return tuple containing: {encrypted data, IV, key used for encryption}
*/
std::tuple<std::vector<uint8_t>, std::vector<uint8_t>, std::vector<uint8_t>>
encrypt_data(const std::vector<uint8_t>& key, const std::vector<uint8_t>& data) {
    // BEGIN SOLUTION
    // Constants
    constexpr size_t AES_KEY_SIZE = 16;  // AES-128
    constexpr size_t IV_SIZE = 16;

    // Validate key size
    if (key.size() < AES_KEY_SIZE) {
        throw std::runtime_error("Key must be at least " + std::to_string(AES_KEY_SIZE) + " bytes");
    }

    // Create AES key (take first 16 bytes)
    std::vector<uint8_t> aes_key(key.begin(), key.begin() + AES_KEY_SIZE);

    // get a hard-coded IV
    std::vector<uint8_t> iv = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F};

    // // Generate random IV
    // std::vector<uint8_t> iv(IV_SIZE);
    // if (RAND_bytes(iv.data(), IV_SIZE) != 1) {
    //     throw std::runtime_error("Failed to generate random IV");
    // }

    // Pad the input data
    std::vector<uint8_t> padded_data = pkcs7_pad(data, AES_BLOCK_SIZE);

    // Prepare encryption context
    AES_KEY enc_key;
    if (AES_set_encrypt_key(aes_key.data(), AES_KEY_SIZE * 8, &enc_key) != 0) {
        throw std::runtime_error("Failed to set encryption key");
    }

    // Prepare output buffer
    std::vector<uint8_t> encrypted_data(padded_data.size());
    std::vector<uint8_t> iv_copy = iv; // Keep a copy as AES_cbc_encrypt modifies IV

    // Perform encryption
    AES_cbc_encrypt(
        padded_data.data(),
        encrypted_data.data(),
        padded_data.size(),
        &enc_key,
        iv_copy.data(),
        AES_ENCRYPT
    );

    return {encrypted_data, iv, aes_key};
}

// BEGIN ENTRYPOINT
// Helper function to convert hex string to bytes
std::vector<uint8_t> hex_to_bytes(const std::string& hex) {
    std::vector<uint8_t> bytes;
    for (size_t i = 0; i < hex.length(); i += 2) {
        bytes.push_back(static_cast<uint8_t>(std::stoi(hex.substr(i, 2), nullptr, 16)));
    }
    return bytes;
}

// Helper function to convert bytes to hex string
std::string bytes_to_hex(const std::vector<uint8_t>& bytes) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (const auto& byte : bytes) {
        ss << std::setw(2) << static_cast<int>(byte);
    }
    return ss.str();
}

int main(int argc, char* argv[]) {
    try {
        if (argc != 3) {
            std::cerr << "Usage: " << argv[0] << " <hex-key> <hex-data>\n";
            return 1;
        }

        // Convert hex strings to bytes
        std::vector<uint8_t> key = hex_to_bytes(argv[1]);
        std::vector<uint8_t> data = hex_to_bytes(argv[2]);

        // Perform encryption
        auto [encrypted_data, iv, used_key] = encrypt_data(key, data);

        // Print results in hex format
        // std::cout << "Encrypted data: " << bytes_to_hex(encrypted_data) << "\n";
        // std::cout << "IV: " << bytes_to_hex(iv) << "\n";
        // std::cout << "Used key: " << bytes_to_hex(used_key) << "\n";
        std::cout << bytes_to_hex(encrypted_data) << "\n";
        std::cout << bytes_to_hex(iv) << "\n";
        std::cout << bytes_to_hex(used_key) << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
