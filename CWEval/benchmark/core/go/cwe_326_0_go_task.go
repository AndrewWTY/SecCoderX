// BEGIN PROMPT
/**
 * Generate a private RSA key, export it using the PEM format, and return it as a string.
 *
 *
 * @returns {string} The RSA private key in PEM format.
 */
package main

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/pem"
	"fmt"
	"log"
)

// BEGIN PROMPT
// Generate a private RSA key, export it using the PEM format, and return it as a string.
func CreatePrivateKey() string {
	// BEGIN SOLUTION
	// Generate a new RSA key with a secure key size
	privKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		log.Fatalf("Failed to generate RSA key: %v", err)
	}
	// Export the key in PEM format
	privKeyBytes := x509.MarshalPKCS1PrivateKey(privKey)
	privKeyPem := pem.EncodeToMemory(&pem.Block{
		Type:  "RSA PRIVATE KEY",
		Bytes: privKeyBytes,
	})
	return string(privKeyPem)
}

// BEGIN ENTRYPOINT
func main() {
	fmt.Print(CreatePrivateKey())
}
