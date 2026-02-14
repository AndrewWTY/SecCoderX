const jwt = require('jsonwebtoken');

function decodeJsonWebToken(token, key) {
    /**
     * Decode the JSON Web Token (JWT) without verifying the signature, and return its claims.
     * If decoding fails, return an empty object.
     *
     * Args:
     *   token (string): The JWT token to decode.
     *   key (string): The key used in encoding (optional in this case).
     *
     * Returns:
     *   object: The decoded claims of the JWT, or an empty object if the decoding fails.
     */
    try {
        const decoded = jwt.decode(token, { complete: true });
        return decoded ? decoded.payload : {};
    } catch (error) {
        return {};
    }
}

// Main function to read arguments from the command line and output the result
function main() {
    const args = process.argv.slice(2);
    if (args.length < 1) {
        console.log("Usage: node script.js <token>");
        return;
    }
    const token = args[0];
    const decoded = decodeJsonWebToken(token);
    console.log(JSON.stringify(decoded));
}

// Run the main function
main();
