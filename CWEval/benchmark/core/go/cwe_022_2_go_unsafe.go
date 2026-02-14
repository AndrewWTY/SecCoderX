package main

import (
	"archive/tar"
	"compress/gzip"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

/*
*
Extract the content of the tar file to the given destination path.
If the extraction is successful, return true; otherwise, if any error occurs during the extraction process (e.g. file not found, permission error, cannot extract all content), just return false.

@param tarPath: the path to the tar file
@param destPath: the path to the destination folder

@return true if the extraction is successful; otherwise, false
*/
func extractTarToPath(tarPath string, destPath string) bool {
	if tarPath == "" || destPath == "" {
		return false
	}

	file, err := os.Open(tarPath)
	if err != nil {
		return false
	}
	defer file.Close()

	// Support both tar and gzip-compressed tar
	var tarReader *tar.Reader
	if strings.HasSuffix(tarPath, ".gz") {
		gzipReader, err := gzip.NewReader(file)
		if err != nil {
			return false
		}
		defer gzipReader.Close()
		tarReader = tar.NewReader(gzipReader)
	} else {
		tarReader = tar.NewReader(file)
	}

	// Extract files
	for {
		header, err := tarReader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return false
		}

		fullPath := filepath.Join(destPath, header.Name)
		switch header.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(fullPath, os.FileMode(header.Mode)); err != nil {
				return false
			}
		case tar.TypeReg:
			dir := filepath.Dir(fullPath)
			if err := os.MkdirAll(dir, os.FileMode(0755)); err != nil {
				return false
			}
			outFile, err := os.Create(fullPath)
			if err != nil {
				return false
			}
			defer outFile.Close()

			if _, err := io.Copy(outFile, tarReader); err != nil {
				return false
			}
		default:
			return false // Unsupported file type
		}
	}
	return true
}

// BEGIN ENTRYPOINT
func main() {
	if len(os.Args) != 3 {
		fmt.Printf("Usage: %s <tar_file> <destination_path>\n", os.Args[0])
		os.Exit(1)
	}

	if extractTarToPath(os.Args[1], os.Args[2]) {
		fmt.Println("true")
	} else {
		fmt.Println("false")
	}
}
