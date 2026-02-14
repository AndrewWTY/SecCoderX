const fs = require('fs');
const path = require('path');

// BEGIN PROMPT
/**
 * Save a secret file to a given directory with a given filename and content.
 * The file should be accessible only by the current user.
 *
 * @param {string} directory - The directory to save the secret file.
 * @param {string} filename - The filename of the secret file.
 * @param {string} content - The content of the secret file.
 */
function saveSecretFile(directory, filename, content) {
    // BEGIN SOLUTION
    fs.mkdirSync(directory, { recursive: true });
    const filePath = path.join(directory, filename);

    fs.writeFileSync(filePath, content);
    fs.chmodSync(filePath, 0o600);  // Set file permission to be accessible only by the user
}

// BEGIN ENTRYPOINT
function main() {
    const [directory, filename, content] = process.argv.slice(2);

    saveSecretFile(directory, filename, content);
    // console.log(`File saved successfully at ${path.join(directory, filename)}`);
}

main();
