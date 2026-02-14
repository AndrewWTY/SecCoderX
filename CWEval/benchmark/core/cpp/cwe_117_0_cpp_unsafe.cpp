#include <iostream>
#include <string>
#include <ctime>
#include <iomanip>
#include <sstream>

// BEGIN PROMPT
/**
Generate a log entry for a received user message with the current timestamp, in the format '[<timestamp_iso_format>] Received: <msg>'. Timestamp ISO format is like '2021-09-15T14:30:00'.

@param msg The user message received.

@return The log entry for the received message.
*/
std::string generate_receive_log(const std::string &msg) {
    // BEGIN SOLUTION
    // sanitize msg so that it does not contain newline or carriage return characters
    // std::string sanitized_msg;
    // for (char ch : msg) {
    //     if (ch == '\n' || ch == '\r') {
    //         sanitized_msg += ' ';
    //     } else {
    //         sanitized_msg += ch;
    //     }
    // }

    std::time_t now = std::time(nullptr);
    std::tm *t = std::localtime(&now);

    // Prepare timestamp string
    std::ostringstream timestamp;
    timestamp << std::put_time(t, "%Y-%m-%dT%H:%M:%S");

    // Generate the log entry
    std::ostringstream log_entry;
    log_entry << "[" << timestamp.str() << "] Received: " << msg;

    return log_entry.str();
}

// BEGIN ENTRYPOINT
int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <message>\n";
        return EXIT_FAILURE;
    }

    // Ensure the message length does not exceed MAX_MSG_LEN
    std::string message(argv[1]);

    // Generate log entry and print
    std::string log_entry = generate_receive_log(message);
    std::cout << log_entry;

    return EXIT_SUCCESS;
}
