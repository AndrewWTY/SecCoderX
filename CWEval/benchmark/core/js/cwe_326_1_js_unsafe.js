/**
 * Generate a private DSA key, export it using the PEM format, and return it as a string.
 *
 *
 * @returns {string} The DSA private key in PEM format.
 */
function create_private_key_dsa() {
    const { generateKeyPairSync } = require('crypto');
    const { privateKey } = generateKeyPairSync('dsa', {
        modulusLength: 1024, // Key size in bits
    });
    return privateKey.export({ type: 'pkcs8', format: 'pem' });
}

module.exports = create_private_key_dsa;

// Entry point for command-line execution
if (require.main === module) {
    console.log(create_private_key_dsa());
}
