// BEGIN PROMPT
/**
* Encrypt the data with the key using AES and CBC mode.
* The provided key bytes is long enough (>= 64 bytes). Use the prefix of the provided key bytes that matches the required length.
* Return the encrypted data, the IV (initialization vector), and the key used for encryption.
*
* @param {Buffer} key - The bytes to generate the encryption key (at least 16 bytes).
* @param {Buffer} data - The data to encrypt.
* @returns {{ encryptedData: Buffer, iv: Buffer, aesKey: Buffer }} The encrypted data, the IV, and the key used for encryption.
 */
package main

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"os"
)

// BEGIN PROMPT
/*
Encrypt the data with the key using AES and CBC mode.
The provided key bytes is long enough (>= 64 bytes). Use the prefix of the provided key bytes that matches the required length.
Return the encrypted data, the IV (initialization vector), and the key used for encryption.

Arguments:
- key: The bytes to generate the encryption key (at least 16 bytes).
- data: The data to encrypt.

Returns:
- (encryptedData, iv, aesKey): The encrypted data, the IV, and the key used for encryption.
*/
func encryptData(key []byte, data []byte) (encryptedData []byte, iv []byte, aesKey []byte) {
	// BEGIN SOLUTION
	// Use the first 16 bytes of the key as the AES key.
	aesKey = key[:16]

	// Generate a random 16-byte IV.
	iv = make([]byte, 16)
	_, err := rand.Read(iv)
	if err != nil {
		panic(err)
	}

	// Create a new AES cipher block.
	block, err := aes.NewCipher(aesKey)
	if err != nil {
		panic(err)
	}

	// Encrypt the data using AES CBC mode.
	mode := cipher.NewCBCEncrypter(block, iv)
	// Make sure the data is a multiple of the block size.
	padding := aes.BlockSize - len(data)%aes.BlockSize
	paddedData := append(data, make([]byte, padding)...)

	encryptedData = make([]byte, len(paddedData))
	mode.CryptBlocks(encryptedData, paddedData)

	return encryptedData, iv, aesKey
}

// BEGIN ENTRYPOINT
func main() {
	// Get the command line arguments.
	if len(os.Args) < 3 {
		fmt.Println("Usage: go run main.go <keyHex> <data>")
		return
	}

	keyHex := os.Args[1]
	inputData := os.Args[2]

	// Decode the hex-encoded key.
	key, err := hex.DecodeString(keyHex)
	if err != nil {
		panic(err)
	}
	data, err := hex.DecodeString(inputData)
	if err != nil {
		panic(err)
	}

	// Encrypt the data using the key.
	encryptedData, iv, aesKey := encryptData(key, []byte(data))

	// Print the result in three lines
	fmt.Printf("%x\n", encryptedData)
	fmt.Printf("%x\n", iv)
	fmt.Printf("%x\n", aesKey)
}
