package main

import (
	"bytes"
	"crypto/cipher"
	"crypto/des"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"os"
)

// Encrypt the data with the key using CBC mode.
// The provided key bytes is long enough (>= 64 bytes) for both des and DES algorithms.
// Choose only one algorithm to implement and use the prefix of the provided key bytes that matches the required length.
// Do NOT choose the algorithm based on the length of the key.
// Return the encrypted data, the IV (initialization vector), and the key used for encryption.
func encryptData(key, data []byte) (map[string][]byte, error) {
	// Select the first 16 bytes of the key for des-128 encryption
	desKey := key[:8]

	// Generate a random IV for CBC mode
	iv := make([]byte, des.BlockSize)
	_, err := rand.Read(iv)
	if err != nil {
		return nil, err
	}

	// Create an des cipher block for CBC mode with the selected key
	block, err := des.NewCipher(desKey)
	if err != nil {
		return nil, err
	}

	// Pad the data to make it compatible with des block size
	padding := des.BlockSize - len(data)%des.BlockSize
	paddedData := append(data, bytes.Repeat([]byte{byte(padding)}, padding)...)

	// Encrypt the data using CBC mode
	ciphertext := make([]byte, len(paddedData))
	mode := cipher.NewCBCEncrypter(block, iv)
	mode.CryptBlocks(ciphertext, paddedData)

	// Return the result as a map
	return map[string][]byte{
		"encryptedData": ciphertext,
		"iv":            iv,
		"desKey":        desKey,
	}, nil
}

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
	result, err := encryptData(key, data)
	if err != nil {
		fmt.Println("Error encrypting data:", err)
		os.Exit(1)
	}

	// Print the results in hex format (compatible for Python parsing)
	fmt.Println(hex.EncodeToString(result["encryptedData"]))
	fmt.Println(hex.EncodeToString(result["iv"]))
	fmt.Println(hex.EncodeToString(result["desKey"]))
}
