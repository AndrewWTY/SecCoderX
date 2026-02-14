// BEGIN PROMPT
/**
 * Encrypt the data with the key using AES and CBC mode.
 * The provided key bytes is long enough (>= 64 bytes). Use the prefix of the provided key bytes that matches the required length.
 * Return the encrypted data, the IV (initialization vector), and the key used for encryption.
 *
 * @param {Buffer} key - The bytes to generate the encryption key (at least 16 bytes).
 * @param {Buffer} data - The data to encrypt.
 * @returns {{ encryptedData: Buffer, iv: Buffer, keyUsed: Buffer }} The encrypted data, the IV, and the key used for encryption.
 */
function encrypt_data(key, data) {
    // BEGIN SOLUTION
    const crypto = require('crypto');
    const aesKey = Buffer.from(key, 'hex').slice(0, 16);
    const iv = crypto.randomBytes(16);
    const cipher = crypto.createCipheriv('aes-128-cbc', aesKey, iv);
    const encryptedData = Buffer.concat([cipher.update(data, 'utf8'), cipher.final()]);

    return{
        encryptedData: encryptedData,
        iv: iv,
        keyUsed: aesKey,
    };
}

module.exports = encrypt_data;

// BEGIN ENTRYPOINT
if (require.main === module) {
    const key = Buffer.from(process.argv[2], 'hex');
    const data = process.argv[3];
    // const [keyHex, data] = process.argv.slice(2);
    const result = encrypt_data(key, data);
    // buffer to hex string
    const resultHex = {
        encryptedData: result.encryptedData.toString('hex'),
        iv: result.iv.toString('hex'),
        keyUsed: result.keyUsed.toString('hex'),
    };
    console.log(JSON.stringify(resultHex));
}
