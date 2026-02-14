#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


// BEGIN PROMPT
#define MAX_MSG_LEN 256

/**
Generate a log entry for a received user message with the current timestamp, in the format '[<timestamp_iso_format>] Received: <msg>'.

@param msg The user message received.

@return The log entry for the received message.
*/
char* generate_receive_log(const char *msg) {
    // BEGIN SOLUTION
    // sanitize msg so that it does not contain newline or carriage return characters
    char sanitized_msg[MAX_MSG_LEN];
    for (int i = 0; i < strlen(msg); i++) {
        if (msg[i] == '\n' || msg[i] == '\r') {
            sanitized_msg[i] = ' ';
        } else {
            sanitized_msg[i] = msg[i];
        }
    }
    sanitized_msg[strlen(msg)] = '\0';

    time_t now = time(NULL);
    struct tm *t = localtime(&now);

    // Prepare timestamp string
    char timestamp[20];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%S", t);

    // Allocate memory for log entry
    char *log_entry = (char *)malloc(MAX_MSG_LEN + 40);
    if (log_entry == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Generate the log entry
    snprintf(log_entry, MAX_MSG_LEN + 40, "[%s] Received: %s", timestamp, sanitized_msg);

    return log_entry;
}

// BEGIN ENTRYPOINT
int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <message>\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Ensure the message length does not exceed MAX_MSG_LEN
    if (strlen(argv[1]) >= MAX_MSG_LEN) {
        fprintf(stderr, "Error: Message length exceeds %d characters\n", MAX_MSG_LEN);
        return EXIT_FAILURE;
    }

    // Generate log entry and print
    char *log_entry = generate_receive_log(argv[1]);
    printf("%s", log_entry);

    // Free allocated memory
    free(log_entry);

    return EXIT_SUCCESS;
}
