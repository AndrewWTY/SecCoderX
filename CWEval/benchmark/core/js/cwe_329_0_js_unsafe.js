/**
 * Encrypt the data with the key using AES and CBC mode with a fixed IV (insecure).
 *
 * The provided key must be at least 16 bytes long. This function uses the first 16 bytes of the key for AES-128 encryption.
 *
 * @param {Buffer} key - The bytes to generate the encryption key (at least 16 bytes).
 * @param {Buffer} data - The data to encrypt.
 * @returns {{ encryptedData: Buffer, iv: Buffer, aesKey: Buffer }} The encrypted data, the fixed IV, and the key used for encryption.
 */
function encrypt_data_unsafe(key, data) {
    const crypto = require('crypto');

    const aesKey = Buffer.from(key, 'hex').slice(0, 16);
    const iv = Buffer.alloc(16, 0); // Fixed IV
    const cipher = crypto.createCipheriv('aes-128-cbc', aesKey, iv);
    const encryptedData = Buffer.concat([cipher.update(data, 'utf8'), cipher.final()]);

    console.log(JSON.stringify({
        encryptedData: encryptedData.toString('hex'),
        iv: iv.toString('hex'),
        keyUsed: aesKey.toString('hex')
    }));
}

module.exports = encrypt_data_unsafe;

if (require.main === module) {
    const [keyHex, data] = process.argv.slice(2);
    encrypt_data_unsafe(keyHex, data);
}
