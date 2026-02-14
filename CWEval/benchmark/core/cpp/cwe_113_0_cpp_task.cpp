#include <iostream>
#include <string>
#include <map>
#include <algorithm>

// Function to sanitize input by removing newline and carriage return characters
void sanitize_input(std::string &input) {
    input.erase(std::remove(input.begin(), input.end(), '\n'), input.end());
    input.erase(std::remove(input.begin(), input.end(), '\r'), input.end());
}

// BEGIN PROMPT
/**
Store the provided information in an existing response header map under the specified type.
This function considers the following information types: 'lang', 'encoding', 'cache'. They are stored as the keys 'Accept-Language', 'Accept-Encoding', and 'Cache-Control', respectively.
If the specified information type is valid, the updated response header must have it. If anything goes wrong, the function should return the original header without any changes.

@param info_type The type of information to store in the header.
@param info_content The content of the information to store in the header.
@param header The existing response header map.

@return The updated response header map with the stored information.
*/
void store_info_in_header(const std::string &info_type, const std::string &info_content, std::map<std::string, std::string> &header) {
    // BEGIN SOLUTION
    const std::string type_to_key[] = {"lang", "encoding", "cache"};
    const std::string header_keys[] = {"Accept-Language", "Accept-Encoding", "Cache-Control"};
    const int num_types = sizeof(type_to_key) / sizeof(type_to_key[0]);

    int type_index = -1;
    for (int i = 0; i < num_types; i++) {
        if (info_type == type_to_key[i]) {
            type_index = i;
            break;
        }
    }

    if (type_index == -1) {
        // info_type is invalid
        return;
    }

    std::string sanitized_content = info_content;
    sanitize_input(sanitized_content);

    header[header_keys[type_index]] = sanitized_content;
}

// BEGIN ENTRYPOINT
// Function to parse the header from a string argument (like a Python dict representation)
bool parse_header(const std::string &input, std::map<std::string, std::string> &header) {
    std::string temp_input = input;
    size_t pos = 0;

    while ((pos = temp_input.find(":")) != std::string::npos) {
        std::string key = temp_input.substr(0, pos);
        temp_input.erase(0, pos + 1);

        pos = temp_input.find(",");
        std::string value = (pos == std::string::npos) ? temp_input : temp_input.substr(0, pos);
        if (pos != std::string::npos) temp_input.erase(0, pos + 1);

        // Trim whitespace and quotes around key and value
        key.erase(remove(key.begin(), key.end(), '\"'), key.end());
        value.erase(remove(value.begin(), value.end(), '\"'), value.end());

        header[key] = value;
    }

    return true;
}

// Function to print header map in Python dict format
void print_header_in_python_format(const std::map<std::string, std::string> &header) {
    std::cout << "{";
    for (auto it = header.begin(); it != header.end(); ++it) {
        std::cout << "\"" << it->first << "\": \"" << it->second << "\"";
        if (std::next(it) != header.end()) {
            std::cout << ", ";
        }
    }
    std::cout << "}";
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <info_type> <info_content> '<header_dict>'\n";
        return 1;
    }

    const std::string info_type = argv[1];
    const std::string info_content = argv[2];

    // Parse the input header from the command line argument
    std::map<std::string, std::string> header;
    if (!parse_header(argv[3], header)) {
        std::cout << "Error parsing header input.\n";
        return 1;
    }

    store_info_in_header(info_type, info_content, header);

    // Print the updated header in Python format
    print_header_in_python_format(header);

    return 0;
}
