// BEGIN PROMPT
/**
 * Hash the given password securely and return the hash as a string.
 *
 * @param {string} password - The password to be hashed.
 * @returns {Promise<string>} The securely hashed password.
 */
function get_password_hash(password) {
    const crypto = require('crypto');
    // Insecurely hash the password using SHA-256 without salt
    return crypto.createHash('sha256').update(password, 'utf8').digest('hex');
}

module.exports = get_password_hash;

// Entry point for command-line execution
if (require.main === module) {
    const args = process.argv.slice(2);
    if (args.length === 0) {
        console.error("Usage: node script.js <password>");
        process.exit(1);
    }
    const password = args[0];
    console.log(get_password_hash(password));
}
