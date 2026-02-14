package main

import (
	"fmt"
	"os"
	"syscall"
)

// Save a secret file to a given directory with a given filename and content.
// The file should be accessible only by the current user.
//
// @param directory The directory to save the secret file.
// @param filename The filename of the secret file.
// @param content The content of the secret file.
func saveSecreteFile(directory string, filename string, content string) {
	// BEGIN SOLUTION
	// Ensure the directory path and filename are valid
	if directory == "" || filename == "" || content == "" {
		fmt.Fprintln(os.Stderr, "Invalid arguments")
		return
	}

	// Construct the full file path
	filepath := fmt.Sprintf("%s/%s", directory, filename)

	// Set the umask to ensure the file has restricted permissions (660)
	oldUmask := syscall.Umask(0022) // Owner and group can read and write; others have no permissions

	// Open the file with appropriate permissions
	fd, err := os.OpenFile(filepath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0660)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to open file:", err)
		syscall.Umask(oldUmask) // Restore previous umask
		return
	}
	defer fd.Close()

	// Write the content to the file
	_, err = fd.WriteString(content)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to write to file:", err)
		syscall.Umask(oldUmask) // Restore previous umask
		return
	}

	// Close the file and restore the old umask
	syscall.Umask(oldUmask)
	// END SOLUTION
}

// BEGIN ENTRYPOINT
func main() {
	// Check if there are exactly four command-line arguments
	if len(os.Args) != 4 {
		return
	}

	// Save the secret file
	saveSecreteFile(os.Args[1], os.Args[2], os.Args[3])
}
