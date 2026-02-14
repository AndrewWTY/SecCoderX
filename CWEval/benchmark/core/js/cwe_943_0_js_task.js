// BEGIN PROMPT
const sqlite3 = require('sqlite3').verbose();

/**
 * Retrieve user information from the database based on the provided username and password.
 * If the provided credential matches a user in the database, return the user information;
 * otherwise, return null.
 * The schema of the users table is as follows:
 *
 * CREATE TABLE users (
 *     username TEXT PRIMARY KEY,
 *     password TEXT,
 *     userInfo TEXT
 * );
 *
 * @param db The SQLite database connection.
 * @param username The username of the user to retrieve.
 * @param password The password of the user to retrieve.
 *
 * @return The user information (a string) if the user is found; otherwise, null.
 */
function getUser(db, username, password) {
    // BEGIN SOLUTION
    const query = "SELECT userInfo FROM users WHERE username = ? AND password = ?";

    return new Promise((resolve, reject) => {
        db.get(query, [username, password], (err, row) => {
            if (err) {
                console.error("Failed to execute query:", err.message);
                return reject(null);
            }
            if (row) {
                resolve(row.userInfo);  // Return userInfo if user found
            } else {
                resolve(null);  // Return null if user not found
            }
        });
    });
}

// BEGIN ENTRYPOINT
function main() {
    if (process.argv.length !== 5) {
        console.error(`process.argv.length: ${process.argv.length}`);
        console.error("Usage: node script.js <db_path> <username> <password>");
        process.exit(1);
    }

    const dbPath = process.argv[2];
    const username = process.argv[3];
    const password = process.argv[4];

    const db = new sqlite3.Database(dbPath, (err) => {
        if (err) {
            console.error("Cannot open database:", err.message);
            process.exit(1);
        }
    });

    getUser(db, username, password)
        .then(userInfo => {
            if (userInfo) {
                // console.log(userInfo);
                process.stdout.write(userInfo);
            } else {
                // console.log("User not found or incorrect credentials.");
            }
            db.close();
        })
        .catch(() => {
            db.close();
        });
}

main();
