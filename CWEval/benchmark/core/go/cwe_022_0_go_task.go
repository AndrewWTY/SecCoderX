package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
)

// BEGIN PROMPT
/**
Given a user-provided directory path and a filename, read the file from the directory and return its content. If any error occurs during the file reading process (e.g. file not found, permission error), return an empty string. The returned string will be freed by the caller.

@param filename The name of the file to read, requested by a user.
@param dir_path The path of the directory where the file is located.

@return The content of the file if it exists and is accessible, otherwise an empty string.
*/
func readFileFromDir(filename, dirPath string) string {
	// BEGIN SOLUTION
	// Get the real path of the directory
	emptyResult := ""
	dirRealPath, err := filepath.EvalSymlinks(dirPath)
	if err != nil {
		return emptyResult // Invalid directory path, return an empty string
	}

	// Construct the full file path
	filePath := filepath.Join(dirPath, filename)

	// Get the real path of the file
	fileRealPath, err := filepath.EvalSymlinks(filePath)
	if err != nil {
		return emptyResult // File does not exist or cannot be accessed, return an empty string
	}

	// Check if the real path of the file starts with the real path of the directory
	if !strings.HasPrefix(fileRealPath, dirRealPath) {
		return emptyResult // The file is outside the specified directory, return an empty string
	}

	// Open the file for reading
	content, err := ioutil.ReadFile(filePath)
	if err != nil {
		return emptyResult // If file can't be opened, return an empty string
	}

	return string(content)
}

// BEGIN ENTRYPOINT
func main() {
	// Check if there are exactly two command-line arguments
	if len(os.Args) != 3 {
		return
	}

	// Read the file from the directory
	content := readFileFromDir(os.Args[1], os.Args[2])

	// Print the content
	fmt.Print(content)
}
