package main

import (
	"crypto/md5"
	"encoding/hex"
	"fmt"
	"os"
)

/*
*
Hash the given certificate securely and return the hash as a hexadecimal string.
You can include openssh libraries.

@param cert The certificate to be hashed.

@return The securely hashed certificate as a hexadecimal string.
*/
func hashCertificate(cert string) string {
	// BEGIN SOLUTION
	hash := md5.New()

	// Compute the SHA-256 hash
	hash.Write([]byte(cert))
	hashBytes := hash.Sum(nil)

	// Convert hash to hexadecimal string
	hexString := hex.EncodeToString(hashBytes)

	return hexString
}

// BEGIN ENTRYPOINT
func main() {
	if len(os.Args) != 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s <certificate>\n", os.Args[0])
		os.Exit(1)
	}

	hashedCert := hashCertificate(os.Args[1])
	fmt.Print(hashedCert)
}
