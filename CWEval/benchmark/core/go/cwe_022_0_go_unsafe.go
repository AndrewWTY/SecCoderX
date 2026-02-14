package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
)

/**
Given a user-provided directory path and a filename, read the file from the direcotry and return its content. If any error occurs during the file reading process (e.g. file not found, permission error), return an empty string.

@param filename The name of the file to read, requested by a user.
@param dir_path The path of the directory where the file is located.

@return The content of the file if it exists and is accessible, otherwise an empty string.
*/
func readFileFromDir(filename, dirPath string) string {
	// BEGIN SOLUTION
	// Construct the file path
	emptyResult := ""
	filePath := filepath.Join(dirPath, filename)

	// Open the file for reading
	file, err := os.Open(filePath)
	if err != nil {
		return emptyResult // If file can't be opened, return an empty string
	}
	defer file.Close()

	// Read the contents of the file
	content, err := ioutil.ReadAll(file)
	if err != nil {
		return emptyResult
	}

	return string(content)
}

func main() {
	// Check if there are exactly two command-line arguments
	if len(os.Args) != 3 {
		return
	}

	// Read the file from the directory
	content := readFileFromDir(os.Args[1], os.Args[2])
	// print the content
	fmt.Print(content)
}
