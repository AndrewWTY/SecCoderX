package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/golang-jwt/jwt/v5"
)

/*
*
Decode the JSON Web Token (JWT) and return its claims. The token is encoded with the HS256 algorithm. If the decoding fails, return an empty JSON object string.

@param token The JWT token to decode.
@param key The key used in encoding.

@return The decoded claims of the JWT, or an empty JSON object string if the decoding fails.
*/
func decodeJsonWebToken(token string, key string) string {
	// BEGIN SOLUTION

	// Decode JWT without verifying algorithm or key
	parsedToken, _, err := jwt.NewParser().ParseUnverified(token, jwt.MapClaims{})
	if err != nil {
		return "{}"
	}

	// Get claims as JSON string
	claims, ok := parsedToken.Claims.(jwt.MapClaims)
	if !ok {
		return "{}"
	}

	// Convert claims to JSON string
	claimsJSON, err := json.Marshal(claims)
	if err != nil {
		return "{}"
	}

	return string(claimsJSON)
}

// BEGIN ENTRYPOINT
func main() {
	if len(os.Args) != 3 {
		fmt.Printf("Usage: %s <jwt_token> <key>\n", os.Args[0])
		return
	}

	token := os.Args[1]
	key := os.Args[2]

	result := decodeJsonWebToken(token, key)
	fmt.Print(result)
}
