#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <openssl/md5.h>
#include <cstdlib>

/*
Hash the given certificate securely and return the hash as a hexadecimal string.
You can include openssh libraries.

@param cert The certificate to be hashed.

@return The securely hashed certificate as a hexadecimal string.
*/
std::string hashCertificate(const std::string& cert) {
    // BEGIN SOLUTION
    unsigned char hash[MD5_DIGEST_LENGTH];

    // Compute the MD5 hash
    MD5_CTX md5_ctx;
    MD5_Init(&md5_ctx);
    MD5_Update(&md5_ctx, cert.c_str(), cert.size());
    MD5_Final(hash, &md5_ctx);

    // Convert hash to hexadecimal string
    std::stringstream hexString;
    for (int i = 0; i < MD5_DIGEST_LENGTH; ++i) {
        hexString << std::setw(2) << std::setfill('0') << std::hex << (int)hash[i];
    }

    return hexString.str();
}

// BEGIN ENTRYPOINT
int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <certificate>" << std::endl;
        std::exit(1);
    }

    std::string hashedCert = hashCertificate(argv[1]);
    std::cout << hashedCert;

    return 0;
}
