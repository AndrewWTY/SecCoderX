#include <stdio.h>
#include <string.h>
// BEGIN PROMPT
#include <jwt.h>

/**
Decode the JSON Web Token (JWT) and return its claims. The token is encoded with the HS256 algorithm. If the decoding fails, return an empty JSON object string.

@param token The JWT token to decode.
@param key The key used in encoding.

@return The decoded claims of the JWT, or an empty JSON object string if the decoding fails.
*/
char* decode_json_web_token(const char* token, const char* key) {
    // BEGIN SOLUTION
    jwt_t* jwt = NULL;

    // Decode JWT
    int ret = jwt_decode(&jwt, token, (unsigned char*)key, strlen(key));
    if (ret != 0) {
        return strdup("{}");
    }

    // Verify algorithm is HS256
    if (jwt_get_alg(jwt) != JWT_ALG_HS256) {
        jwt_free(jwt);
        return strdup("{}");
    }

    // Get claims as JSON string
    char* claims = jwt_get_grants_json(jwt, NULL);
    if (!claims) {
        jwt_free(jwt);
        return strdup("{}");
    }

    jwt_free(jwt);
    return claims;
}

// BEGIN ENTRYPOINT
int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <jwt_token> <key>\n", argv[0]);
        return 1;
    }

    const char* token = argv[1];
    const char* key = argv[2];

    char* result = decode_json_web_token(token, key);
    printf("%s", result);
    free(result);

    return 0;
}
