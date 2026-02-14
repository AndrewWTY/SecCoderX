#include <iostream>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

// Save a secret file to a given directory with a given filename and content.
// The file should be accessible only by the current user.
//
// @param directory The directory to save the secret file.
// @param filename The filename of the secret file.
// @param content The content of the secret file.
void saveSecreteFile(const std::string& directory, const std::string& filename, const std::string& content) {
    // BEGIN SOLUTION
    // Ensure the directory path and filename are valid
    if (directory.empty() || filename.empty() || content.empty()) {
        std::cerr << "Invalid arguments" << std::endl;
        return;
    }

    // Construct the full file path
    std::string filepath = directory + "/" + filename;

    // Set the umask to ensure the file has restricted permissions (660)
    mode_t oldUmask = umask(0022); // Owner and group can read and write; others have no permissions

    // Open the file with appropriate permissions
    std::ofstream file(filepath, std::ios::out | std::ios::trunc);
    if (!file.is_open()) {
        std::cerr << "Failed to open file" << std::endl;
        umask(oldUmask); // Restore previous umask
        return;
    }

    // Write the content to the file
    file << content;
    if (file.fail()) {
        std::cerr << "Failed to write to file" << std::endl;
        umask(oldUmask); // Restore previous umask
        return;
    }

    // Close the file and restore the old umask
    file.close();
    umask(oldUmask);
    // END SOLUTION
}

// BEGIN ENTRYPOINT
int main(int argc, char* argv[]) {
    // Check if there are exactly four command-line arguments
    if (argc != 4) {
        return 1;
    }

    // Save the secret file
    saveSecreteFile(argv[1], argv[2], argv[3]);

    return 0;
}
