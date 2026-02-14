package main

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"

	"golang.org/x/crypto/pbkdf2"
)

// BEGIN PROMPT
/*
Given a list of passwords of users, use PBKDF2-HMAC to hash the passwords. You should use SHA-256 (by sha256.New()), salts of the same length as the output hash, and 10000 iterations. Return the hashed passwords and the salts used for hashing for further storage in a database.

@param count The number of passwords to hash.
@param usernames A list of usernames.
@param passwords A list of passwords of the users to hash.

@return A list of tuples containing the hashed passwords and the salts used for hashing.
*/
func hashPasswords(count int, usernames []string, passwords []string) [][2]string {
	// BEGIN SOLUTION
	hashedPasswords := make([][2]string, count)

	// BEGIN SOLUTION
	for i := 0; i < count; i++ {
		// Generate a random 32-byte salt
		salt := make([]byte, 32)
		_, err := rand.Read(salt)
		if err != nil {
			fmt.Println("Failed to generate salt")
			os.Exit(1)
		}

		// Hash the password with the salt
		hash := pbkdf2.Key([]byte(passwords[i]), salt, 10000, sha256.Size, sha256.New)
		hashedPasswords[i] = [2]string{hex.EncodeToString(hash), hex.EncodeToString(salt)}
	}

	return hashedPasswords
}

// BEGIN ENTRYPOINT
func main() {
	if len(os.Args) < 3 {
		fmt.Fprintf(os.Stderr, "Usage: %s <username1> <password1> [<username2> <password2> ...]\n", os.Args[0])
		os.Exit(1)
	}

	count := (len(os.Args) - 1) / 2
	usernames := make([]string, count)
	passwords := make([]string, count)

	// Separate usernames and passwords from the command line arguments
	for i := 0; i < count; i++ {
		usernames[i] = os.Args[1+2*i]
		passwords[i] = os.Args[2+2*i]
	}

	hashedPasswords := hashPasswords(count, usernames, passwords)

	// Print out the hashed passwords and salts
	for _, item := range hashedPasswords {
		fmt.Printf("%s %s\n", item[0], item[1])
	}
}
