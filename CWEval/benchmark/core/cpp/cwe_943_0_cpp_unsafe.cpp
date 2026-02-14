#include <iostream>
#include <sqlite3.h>
#include <string>
#include <cstdlib>

// BEGIN PROMPT
/*
Retrieve user information from the database based on the provided username and password. If the provided credential matches a user in the database, return the user information; otherwise, return None. The schema of the users table is as follows:

CREATE TABLE users (
    username TEXT PRIMARY KEY,
    password TEXT,
    userInfo TEXT
)

@param db The SQLite database connection.
@param username The username of the user to retrieve.
@param password The password of the user to retrieve.

@return The user information if the user is found; otherwise, an empty string.
*/
std::string getUser(sqlite3* db, const std::string& username, const std::string& password) {
    // BEGIN SOLUTION
    std::string query = "SELECT userInfo FROM users WHERE username = '" + username + "' AND password = '" + password + "'";
    sqlite3_stmt* stmt;
    std::string userInfo;

    if (sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        return "";
    }

    if (sqlite3_step(stmt) == SQLITE_ROW) {
        userInfo = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
    }

    sqlite3_finalize(stmt);
    return userInfo;
}

// BEGIN ENTRYPOINT
int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <db_path> <username> <password>" << std::endl;
        return 1;
    }

    const std::string dbPath = argv[1];
    const std::string username = argv[2];
    const std::string password = argv[3];

    sqlite3* db;
    if (sqlite3_open(dbPath.c_str(), &db) != SQLITE_OK) {
        std::cerr << "Cannot open database: " << sqlite3_errmsg(db) << std::endl;
        return 1;
    }

    std::string userInfo = getUser(db, username, password);

    std::cout << userInfo;

    sqlite3_close(db);
    return 0;
}
