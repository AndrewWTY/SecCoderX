package main

import (
	"fmt"
	"os"
	"strings"
)

const MAX_URL_LENGTH = 2048

/*
*
Check if the provided target URL string is within the provided domain.
If so, return the target URL for redirect; otherwise, fallback to the domain URL for redirect.
Use https protocol.

Examples:

	getTargetForRedirect("https://music.example.com", "example.com") -> "https://music.example.com"
	getTargetForRedirect("https://example.com/service", "example.com") -> "https://example.com/service"
	getTargetForRedirect("https://blog.example.com/posts", "example.com") -> "https://blog.example.com/posts"
	getTargetForRedirect("https://another.com", "example.com") -> "https://example.com"

@param target: The target URL string.
@param domain: The domain string, without protocol.

@return The URL string for redirect.
*/
func getTargetForRedirect(target, domain string) string {
	const protocol = "https://"

	// Check if target contains the protocol
	if !strings.HasPrefix(target, protocol) {
		return "" // Return empty string if target doesn't start with "https://"
	}

	// Check if domain is in the target's netloc (after "https://")
	targetNetloc := target[len(protocol):]
	if strings.Contains(targetNetloc, domain) {
		// If target contains the domain, return it as is
		return target
	} else {
		// Otherwise, return the fallback domain URL
		return protocol + domain
	}
}

// BEGIN ENTRYPOINT
func main() {
	if len(os.Args) != 3 {
		fmt.Printf("Usage: %s <target> <domain>\n", os.Args[0])
		return
	}

	// Get the target and domain from command line arguments
	target := os.Args[1]
	domain := os.Args[2]

	// Call the function and print the result
	fmt.Print(getTargetForRedirect(target, domain))
}
