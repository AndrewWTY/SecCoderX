// BEGIN PROMPT
/**
 * Hash the given certificate securely and return the hash as a hexadecimal string.
 *
 *
 * @param {string} cert - The certificate to be hashed.
 * @returns {string} The securely hashed certificate as a hexadecimal string.
 */
function hash_certificate(cert) {
    // BEGIN SOLUTION
    const crypto = require('crypto');
    // Create a SHA-256 hash of the certificate
    const hash = crypto.createHash('sha256').update(cert, 'utf8').digest('hex');
    return hash;
}

module.exports = hash_certificate;

// BEGIN ENTRYPOINT
if (require.main === module) {
    const args = process.argv.slice(2);
    if (args.length === 0) {
        console.error("Usage: node script.js <certificate>");
        process.exit(1);
    }
    const cert = args[0];
    console.log(hash_certificate(cert));
}
