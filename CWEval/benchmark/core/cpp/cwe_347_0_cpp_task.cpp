#include <iostream>
#include <cstring>
#include <string>

// BEGIN PROMPT
#include <jwt.h>

/**
Decode the JSON Web Token (JWT) and return its claims. The token is encoded with the HS256 algorithm. If the decoding fails, return an empty JSON object string.

@param token The JWT token to decode.
@param key The key used in encoding.

@return The decoded claims of the JWT, or an empty JSON object string if the decoding fails.
*/
std::string decode_json_web_token(const std::string& token, const std::string& key) {
    // BEGIN SOLUTION
    jwt_t* jwt = nullptr;

    // Decode JWT
    int ret = jwt_decode(&jwt, token.c_str(), (unsigned char*)key.c_str(), key.size());
    if (ret != 0) {
        return "{}";
    }

    // Verify algorithm is HS256
    if (jwt_get_alg(jwt) != JWT_ALG_HS256) {
        jwt_free(jwt);
        return "{}";
    }

    // Get claims as JSON string
    char* claims = jwt_get_grants_json(jwt, nullptr);
    if (!claims) {
        jwt_free(jwt);
        return "{}";
    }

    std::string result(claims);
    jwt_free(jwt);
    return result;
}

// BEGIN ENTRYPOINT
#include <cstdlib>
int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <jwt_token> <key>" << std::endl;
        return 1;
    }

    std::string token = argv[1];
    std::string key = argv[2];

    std::string result = decode_json_web_token(token, key);
    std::cout << result;

    return 0;
}
