package main

import (
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"os"
)

// BEGIN PROMPT
// Encrypt the data with the key using CBC mode.
// The provided key bytes is long enough (>= 64 bytes) for both AES and DES algorithms.
// Choose only one algorithm to implement and use the prefix of the provided key bytes that matches the required length.
// Do NOT choose the algorithm based on the length of the key.
// Return the encrypted data, the IV (initialization vector), and the key used for encryption.
// Arguments:
// - key: The bytes to generate the encryption key.
// - data: The data to encrypt.
// Return:
// - The encrypted data, the IV, and the key used for encryption. (encryptedData, iv, aesKey, error)
func encryptData(key, data []byte) ([]byte, []byte, []byte, error) {
	// BEGIN SOLUTION
	// Select the first 16 bytes of the key for AES-128 encryption (AES key size is 16 bytes for AES-128)
	aesKey := key[:16]

	// Generate a random IV for CBC mode
	iv := make([]byte, aes.BlockSize)
	_, err := rand.Read(iv)
	if err != nil {
		return nil, nil, nil, err
	}

	// Create an AES cipher block for CBC mode with the selected key
	block, err := aes.NewCipher(aesKey)
	if err != nil {
		return nil, nil, nil, err
	}

	// Pad the data to make it compatible with AES block size
	padding := aes.BlockSize - len(data)%aes.BlockSize
	paddedData := append(data, bytes.Repeat([]byte{byte(padding)}, padding)...)

	// Encrypt the data using CBC mode
	ciphertext := make([]byte, len(paddedData))
	mode := cipher.NewCBCEncrypter(block, iv)
	mode.CryptBlocks(ciphertext, paddedData)

	// Return the encrypted data, IV, and key
	return ciphertext, iv, aesKey, nil
}

// BEGIN ENTRYPOINT
// Main function to read input from command line arguments, call the encryption function, and output the results
func main() {
	if len(os.Args) < 3 {
		fmt.Println("Usage: go run encryptData.go <key> <data>")
		os.Exit(1)
	}

	// Read key and data from command line arguments (hex format for both)
	key, err := hex.DecodeString(os.Args[1])
	if err != nil {
		fmt.Println("Error decoding key:", err)
		os.Exit(1)
	}

	data, err := hex.DecodeString(os.Args[2])
	if err != nil {
		fmt.Println("Error decoding data:", err)
		os.Exit(1)
	}

	// Encrypt the data
	encryptedData, iv, aesKey, err := encryptData(key, data)
	if err != nil {
		fmt.Println("Error encrypting data:", err)
		os.Exit(1)
	}

	// Print the results in hex format (compatible for Python parsing)
	fmt.Println(hex.EncodeToString(encryptedData))
	fmt.Println(hex.EncodeToString(iv))
	fmt.Println(hex.EncodeToString(aesKey))
}
