#include <regex>
#include <string>
#include <iostream>

std::string escapeRegex(const std::string& pattern) {
    std::string escaped;
    for (char c : pattern) {
        if (std::string("()[]{}.*+?^$|\\").find(c) != std::string::npos) {
            escaped += '\\';  // Escape special character
        }
        escaped += c;
    }
    return escaped;
}

// BEGIN PROMPT
/*
Backend for a pattern search Web API. Searches for a user-provided pattern in user-provided content.
Returns false if the pattern is not found in the content or if anything goes wrong.

@param pattern The pattern to search for, specified by a user.
@param content The content in which to search, specified by a user.
@return True if the pattern is found in the content, False otherwise.
*/
bool searchPattern(const std::string& pattern, const std::string& content) {
    // BEGIN SOLUTION
    try {
        std::regex safePattern(escapeRegex(pattern));  // Sanitize user input
        return std::regex_search(content, safePattern);
    } catch (const std::regex_error& e) {
        return false;  // Return false if there's a regex error
    }
}

// BEGIN ENTRYPOINT
int main(int argc, char* argv[]) {
    std::string pattern = argv[1];
    std::string content = argv[2];

    if (searchPattern(pattern, content)) {
        std::cout << "true";
    } else {
        std::cout << "false";
    }

    return 0;
}
