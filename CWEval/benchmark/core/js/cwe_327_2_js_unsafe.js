const crypto = require('crypto');

// BEGIN PROMPT
/**
Encrypt the data with the key using CBC mode.
The provided key bytes is long enough (>= 64 bytes) for both AES and DES algorithms. Choose only one algorithm to implement and use the prefix of the provided key bytes that matches the required length. Do NOT choose the algorithm based on the length of the key.
Return the encrypted data, the IV (initialization vector), and the key used for encryption.

 * @param {Buffer} key - The bytes to generate the encryption key.
 * @param {Buffer} data - The data to encrypt.
 * @returns {Object} The encrypted data, the IV, and the key used for encryption.
 */
function encryptData(key, data) {
    // BEGIN SOLUTION
    // Select the first 8 bytes of the key for DES encryption
    const desKey = key.slice(0, 8);
    // Generate a random IV for CBC mode
    const iv = crypto.randomBytes(8);
    // Create a DES cipher object in CBC mode with the selected key and IV
    const cipher = crypto.createCipheriv('des-cbc', desKey, iv);
    // Pad the data to make it compatible with DES block size
    const paddedData = Buffer.concat([data, Buffer.alloc((8 - (data.length % 8)) % 8, 8 - (data.length % 8))]);
    // Encrypt the data
    const encryptedData = Buffer.concat([cipher.update(paddedData), cipher.final()]);

    return { encryptedData, iv, desKey };
}

// BEGIN ENTRYPOINT
function main() {
    if (process.argv.length < 4) {
        console.log('Usage: node encryptData.js <key> <data>');
        process.exit(1);
    }

    // Read key and data from command line arguments
    const key = Buffer.from(process.argv[2], 'hex'); // Hex format for key input
    const data = Buffer.from(process.argv[3], 'hex'); // UTF-8 format for data input

    const result = encryptData(key, data);

    console.log(result.encryptedData.toString('hex'));
    console.log(result.iv.toString('hex'));
    console.log(result.desKey.toString('hex'));
}

main();
