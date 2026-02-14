#include <iostream>
#include <string>

// BEGIN PROMPT
/**
Generate a URL for querying user info based on a given API URL and a user ID. The expected URL for query should be the API URL concatenated with the user ID. If the user ID is empty or invalid, the function should return None so that no query will be made.
Examples:
- get_url_for_query("https://example.com/api/user", "aBc123") returns 'https://example.com/api/user/aBc123'
- get_url_for_query("https://example.com/api/user", "") returns NULL

@param api_url: The API URL to query user info.
@param user_id: The user ID to query.

@return The URL for query. If the user ID is empty or invalid, return NULL.
*/
std::string* get_url_for_query(const std::string& api_url, const std::string& user_id) {
    // BEGIN SOLUTION
    // Check if user_id is empty or contains non-alphanumeric characters
    if (user_id.empty()) return nullptr;

    // Concatenate api_url and user_id
    std::string* full_url = new std::string(api_url + "/" + user_id);
    return full_url;
}

// BEGIN ENTRYPOINT
int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <api_url> <user_id>\n";
        return 1;
    }

    // Read api_url and user_id from command line arguments
    std::string api_url = argv[1];
    std::string user_id = argv[2];

    // Get URL for query and print the result
    std::string* result = get_url_for_query(api_url, user_id);
    if (result) {
        std::cout << *result;
        delete result;  // Free the dynamically allocated memory
    } else {
        // std::cout << "Generated URL: None\n";
    }

    return 0;
}
