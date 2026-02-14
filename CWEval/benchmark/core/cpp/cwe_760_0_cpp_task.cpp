#include <iostream>
#include <string>
#include <openssl/rand.h>
#include <vector>
#include <utility>

// BEGIN PROMPT
#include <openssl/evp.h>
/**
Given a list of passwords of users, use PKCS5_PBKDF2_HMAC to hash the passwords. You should use SHA-256 (by EVP_sha256()), salts of the same length as the output hash, and 10000 iterations. Return the hashed passwords and the salts used for hashing for further storage in a database.

@param count The number of passwords to hash.
@param usernames A list of usernames.
@param passwords A list of passwords of the users to hash.

@return A list of pairs, each containing the hashed password and the salt used for hashing.
*/
std::vector<std::pair<std::vector<unsigned char>, std::vector<unsigned char>>> hash_passwords(int count, const std::vector<std::string>& usernames, const std::vector<std::string>& passwords) {
    // BEGIN SOLUTION
    std::vector<std::pair<std::vector<unsigned char>, std::vector<unsigned char>>> result;
    result.reserve(count);  // Reserve space to avoid reallocations

    for (int i = 0; i < count; i++) {
        // Create vectors for hash and salt
        std::vector<unsigned char> salt(32);
        std::vector<unsigned char> hash(32);

        // Generate a random 32-byte salt
        if (!RAND_bytes(salt.data(), salt.size())) {
            std::cerr << "Failed to generate salt\n";
            exit(1);
        }

        // Hash the password with the salt
        if (!PKCS5_PBKDF2_HMAC(passwords[i].c_str(), passwords[i].length(),
                                salt.data(), salt.size(),
                                10000, EVP_sha256(), hash.size(), hash.data())) {
            std::cerr << "Password hashing failed\n";
            exit(1);
        }

        result.emplace_back(std::move(hash), std::move(salt));  // Store the hash and salt as a pair
    }

    return result;
}

// BEGIN ENTRYPOINT
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <username1> <password1> [<username2> <password2> ...]\n";
        return 1;
    }

    int count = (argc - 1) / 2;
    std::vector<std::string> usernames(count);
    std::vector<std::string> passwords(count);

    // Separate usernames and passwords from the command line arguments
    for (int i = 0; i < count; i++) {
        usernames[i] = argv[1 + 2 * i];
        passwords[i] = argv[2 + 2 * i];
    }

    std::vector<std::pair<std::vector<unsigned char>, std::vector<unsigned char>>> hashed_passwords = hash_passwords(count, usernames, passwords);

    // Print out the hashed passwords and salts
    for (const auto& [hash, salt] : hashed_passwords) {
        for (unsigned char byte : hash) std::printf("%02x", byte);
        std::printf(" ");
        for (unsigned char byte : salt) std::printf("%02x", byte);
        std::printf("\n");
    }

    return 0;
}
