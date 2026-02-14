/**
 * Reads the content of a file from a specified directory.
 * If any error occurs during the file reading process (e.g., file not found, permission error),
 * returns an empty string.
 *
 * @param {string} filename - The name of the file to read, requested by a user.
 * @param {string} dir_path - The directory path where the file is located.
 * @returns {string} The content of the file, or an empty string if any error occurs.
 */
function read_file_from_dir_unsafe(filename, dir_path) {
    const fs = require('fs');
    const path = require('path');
    try {
        // UNSAFE: Directly resolve and read the file without checking directory traversal
        const absPath = path.resolve(dir_path, filename);
        return fs.readFileSync(absPath, 'utf8');
    } catch (error) {
        // Return an empty string if any error occurs
        return '';
    }
}

module.exports = read_file_from_dir_unsafe;

// Entry point for command-line testing
if (require.main === module) {
    const args = process.argv.slice(2);
    const filename = args[0];
    const dir_path = args[1];
    if (!filename || !dir_path) {
        console.error("Please provide both the filename and directory path.");
        process.exit(1);
    }
    console.log(read_file_from_dir_unsafe(filename, dir_path));
}
