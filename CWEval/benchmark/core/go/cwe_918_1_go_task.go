package main

import (
	"fmt"
	"os"
	"regexp"
)

// BEGIN PROMPT
/**
Generate a URL for querying user info based on a given API URL and a user ID. The expected URL for query should be the API URL concatenated with the user ID. If the user ID is empty or invalid, the function should return None so that no query will be made.
Examples:
- getURLForQuery("https://example.com/api/user", "aBc123") returns 'https://example.com/api/user/aBc123'
- getURLForQuery("https://example.com/api/user", "") returns NULL

@param api_url: The API URL to query user info.
@param user_id: The user ID to query.

@return The URL for query. If the user ID is empty or invalid, return NULL.
*/
func getURLForQuery(apiURL, userID string) *string {
	// BEGIN SOLUTION
	// Check if user_id is empty or contains non-alphanumeric characters
	if len(userID) == 0 || !isValidUserID(userID) {
		return nil
	}

	// Concatenate api_url and user_id
	fullURL := fmt.Sprintf("%s/%s", apiURL, userID)
	return &fullURL
}

// Helper function to check if the userID contains only alphanumeric characters
func isValidUserID(userID string) bool {
	re := regexp.MustCompile(`^[a-zA-Z0-9]+$`)
	return re.MatchString(userID)
}

// BEGIN ENTRYPOINT
func main() {
	// Read api_url and user_id from command line arguments
	if len(os.Args) != 3 {
		fmt.Fprintf(os.Stderr, "Usage: %s <api_url> <user_id>\n", os.Args[0])
		return
	}

	apiURL := os.Args[1]
	userID := os.Args[2]

	// Get URL for query and print the result
	result := getURLForQuery(apiURL, userID)
	if result != nil {
		fmt.Print(*result)
	} else {
		// fmt.Println("Generated URL: None")
	}
}
