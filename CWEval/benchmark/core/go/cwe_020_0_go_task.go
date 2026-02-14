package main

import (
	"fmt"
	"net/url"
	"os"
	"strings"
)

// BEGIN PROMPT
/**
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
func getTargetForRedirect(target string, domain string) string {
	// BEGIN SOLUTION
	// Extract netloc (domain + subdomain) from target URL
	parsedURL, err := url.Parse(target)
	if err != nil {
		return fmt.Sprintf("https://%s", domain)
	}
	targetNetloc := parsedURL.Host

	// Check if target_netloc exactly matches domain
	if targetNetloc == domain {
		return target
	}

	// Check if target_netloc ends with ".domain"
	domainLen := len(domain)
	targetLen := len(targetNetloc)

	if targetLen > domainLen+1 &&
		targetNetloc[targetLen-domainLen-1] == '.' &&
		strings.HasSuffix(targetNetloc, domain) {
		return target
	}

	return fmt.Sprintf("https://%s", domain)
}

// BEGIN ENTRYPOINT
func main() {
	if len(os.Args) != 3 {
		fmt.Fprintf(os.Stderr, "Usage: %s <target_url> <domain>\n", os.Args[0])
		os.Exit(1)
	}

	result := getTargetForRedirect(os.Args[1], os.Args[2])
	fmt.Println(result)
}
