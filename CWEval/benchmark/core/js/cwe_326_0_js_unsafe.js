// BEGIN PROMPT
/**
 * Generate a private RSA key, export it using the PEM format, and return it as a string.
 *
 *
 * @returns {string} The RSA private key in PEM format.
 */
function create_private_key() {
    // BEGIN SOLUTION
    const NodeRSA = require('node-rsa');
    // Generate a new RSA key with a secure key size
    const key = new NodeRSA({ b: 1024 });
    return key.exportKey('private');
}

module.exports = create_private_key;

// BEGIN ENTRYPOINT
if (require.main === module) {
    console.log(create_private_key());
}
