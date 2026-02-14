package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/golang-jwt/jwt/v5"
)

// BEGIN PROMPT
// Decode the JSON Web Token (JWT) and return its claims. The token is encoded with the HS256 algorithm. If the decoding fails, return an empty JSON object string.
// Use "github.com/golang-jwt/jwt/v5"
//
// @param token The JWT token to decode.
// @param key The key used in encoding.
//
// @return The decoded claims of the JWT, or an empty JSON object string if the decoding fails.
func decodeJsonWebToken(token string, key string) string {
	// BEGIN SOLUTION

	// Parse JWT token
	parsedToken, err := jwt.Parse(token, func(token *jwt.Token) (interface{}, error) {
		// Ensure the algorithm is HS256
		if token.Method != jwt.SigningMethodHS256 {
			return nil, fmt.Errorf("invalid signing method")
		}
		return []byte(key), nil
	})

	if err != nil || !parsedToken.Valid {
		return "{}"
	}

	// Extract claims
	claims := parsedToken.Claims.(jwt.MapClaims)
	claimsJSON, err := json.Marshal(claims)
	if err != nil {
		return "{}"
	}

	// END SOLUTION
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
	fmt.Println(result)
}
