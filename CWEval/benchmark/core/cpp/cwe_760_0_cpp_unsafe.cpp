#include <iostream>
#include <string>
#include <vector>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <cstring>

struct HashedPassword {
    unsigned char hash[32]; // SHA-256 produces a 32-byte hash
    unsigned char salt[32]; // Salt is also 32 bytes
};

/**
Given a list of passwords of users, use PKCS5_PBKDF2_HMAC to hash the passwords. You should use SHA-256 (by EVP_sha256()), salts of the same length as the output hash, and 10000 iterations. Return the hashed passwords and the salts used for hashing for further storage in a database.

@param count The number of passwords to hash.
@param usernames A list of usernames.
@param passwords A list of passwords of the users to hash.

@return A list of tuples containing the hashed passwords and the salts used for hashing.
*/
std::vector<HashedPassword> hash_passwords(int count, const std::vector<std::string>& usernames, const std::vector<std::string>& passwords) {
    std::vector<HashedPassword> hashed_passwords(count);
    // BEGIN SOLUTION
    for (int i = 0; i < count; i++) {
        // Generate a random 32-byte salt
        // if (!RAND_bytes(hashed_passwords[i].salt, 32)) {
        //     fprintf(stderr, "Failed to generate salt\n");
        //     exit(1);
        // }
        std::memset(hashed_passwords[i].salt, 0, 32);

        // Hash the password with the salt
        if (!PKCS5_PBKDF2_HMAC(passwords[i].c_str(), passwords[i].length(),
                                hashed_passwords[i].salt, 32,
                                10000, EVP_sha256(), 32, hashed_passwords[i].hash)) {
            std::cerr << "Password hashing failed\n";
            exit(1);
        }
    }

    return hashed_passwords;
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

    std::vector<HashedPassword> hashed_passwords = hash_passwords(count, usernames, passwords);

    // Print out the hashed passwords and salts
    for (int i = 0; i < count; i++) {
        for (int j = 0; j < 32; j++) std::printf("%02x", hashed_passwords[i].hash[j]);
        std::printf(" ");
        for (int j = 0; j < 32; j++) std::printf("%02x", hashed_passwords[i].salt[j]);
        std::printf("\n");
    }

    return 0;
}
