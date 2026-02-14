#include <stdio.h>
#include <stdlib.h>
#include <sqlite3.h>
#include <string.h>

char *get_user(sqlite3 *db, const char *username, const char *password) {
    sqlite3_stmt *stmt;
    char query[256];
    char *userInfo = NULL;

    snprintf(query, sizeof(query), "SELECT userInfo FROM users WHERE username = '%s' AND password = '%s'", username, password);

    if (sqlite3_prepare_v2(db, query, -1, &stmt, NULL) != SQLITE_OK) {
        fprintf(stderr, "Failed to prepare statement: %s\n", sqlite3_errmsg(db));
        return NULL;
    }

    // Execute the query
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        const unsigned char *result = sqlite3_column_text(stmt, 0);
        userInfo = strdup((const char *)result);  // Copy the result to userInfo
    }

    // Clean up
    sqlite3_finalize(stmt);
    return userInfo;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <db_path> <username> <password>\n", argv[0]);
        return 1;
    }

    const char *username = argv[2];
    const char *password = argv[3];

    sqlite3 *db;
    if (sqlite3_open(argv[1], &db) != SQLITE_OK) {
        fprintf(stderr, "Cannot open database: %s\n", sqlite3_errmsg(db));
        return 1;
    }

    char *userInfo = get_user(db, username, password);

    if (userInfo) {
        printf("%s", userInfo);
        free(userInfo);
    } else {
        // printf("User not found or incorrect credentials.\n");
    }

    sqlite3_close(db);
    return 0;
}
