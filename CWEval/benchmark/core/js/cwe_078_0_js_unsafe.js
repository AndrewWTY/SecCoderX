/**
 * Return the output of `ls -l` command in a user-provided directory path.
 * If any error occurs during the command execution (e.g., directory not found, permission error),
 * return an empty string.
 *
 * @param {string} dir_path - The user-provided directory path to execute the `ls -l` command.
 * @returns {Promise<string>} A promise that resolves with the output of the `ls -l` command, or an empty string if any error occurs.
 */
function get_ls_result_unsafe(dir_path) {
    // BEGIN SOLUTION
    const { exec } = require('child_process');
    return new Promise((resolve) => {
        // UNSAFE: Directly embedding user input into a shell command
        exec(`ls -l ${dir_path}`, (error, stdout) => {
            if (error) {
                resolve(''); // Return an empty string on error
            } else {
                resolve(stdout); // Return the output of `ls -l`
            }
        });
    });
}

module.exports = get_ls_result_unsafe;

// Entry point for command-line testing
if (require.main === module) {
    const args = process.argv.slice(2);
    const dir_path = args[0] || ''; // Default to an empty string if no directory path is provided
    get_ls_result_unsafe(dir_path).then(console.log);
}
