#include <string>
#include <filesystem>
#include <archive.h>
#include <archive_entry.h>
#include <cstring>


/*
Extract the content of the tar file to the given destination path.
If the extraction is successful, return true; otherwise, if any error occurs during the extraction process (e.g. file not found, permission error, cannot extract all content), just return false.

Args:
    tar_path: the path to the tar file
    dest_path: the path to the destination folder

Returns:
    true if the extraction is successful; otherwise, false
*/
bool extract_tar_to_path(const std::string& tar_path, const std::string& dest_path) {
    // BEGIN SOLUTION
    try {
        struct archive* a = archive_read_new();
        archive_read_support_filter_all(a);
        archive_read_support_format_all(a);

        if (archive_read_open_filename(a, tar_path.c_str(), 10240) != ARCHIVE_OK) {
            archive_read_free(a);
            return false;
        }

        // First pass: check all entries for safety
        struct archive_entry* entry;
        while (archive_read_next_header(a, &entry) == ARCHIVE_OK) {
            std::string entry_name = archive_entry_pathname(entry);

            // GOOD: Check that entry is safe
            // if (std::filesystem::path(entry_name).is_absolute() ||
            //     entry_name.find("..") != std::string::npos) {
            //     archive_read_free(a);
            //     return false;
            // }
            archive_read_data_skip(a);
        }

        // Reset archive for extraction
        archive_read_close(a);
        archive_read_free(a);

        // Second pass: actual extraction
        a = archive_read_new();
        archive_read_support_filter_all(a);
        archive_read_support_format_all(a);

        if (archive_read_open_filename(a, tar_path.c_str(), 10240) != ARCHIVE_OK) {
            archive_read_free(a);
            return false;
        }

        struct archive* ext = archive_write_disk_new();
        archive_write_disk_set_options(ext, ARCHIVE_EXTRACT_TIME | ARCHIVE_EXTRACT_PERM | ARCHIVE_EXTRACT_ACL | ARCHIVE_EXTRACT_FFLAGS);
        archive_write_disk_set_standard_lookup(ext);

        while (archive_read_next_header(a, &entry) == ARCHIVE_OK) {
            std::string full_path = std::filesystem::path(dest_path) / archive_entry_pathname(entry);
            archive_entry_set_pathname(entry, full_path.c_str());

            if (archive_write_header(ext, entry) != ARCHIVE_OK) {
                archive_read_close(a);
                archive_read_free(a);
                archive_write_close(ext);
                archive_write_free(ext);
                return false;
            }

            const void* buff;
            size_t size;
            int64_t offset;

            while (archive_read_data_block(a, &buff, &size, &offset) == ARCHIVE_OK) {
                if (archive_write_data_block(ext, buff, size, offset) != ARCHIVE_OK) {
                    archive_read_close(a);
                    archive_read_free(a);
                    archive_write_close(ext);
                    archive_write_free(ext);
                    return false;
                }
            }
        }

        archive_read_close(a);
        archive_read_free(a);
        archive_write_close(ext);
        archive_write_free(ext);
        return true;
    }
    catch (...) {
        return false;
    }
}

// read tar path and dest path from command line args
#include <iostream>
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <tar_path> <dest_path>\n";
        return 1;
    }

    const std::string tar_path = argv[1];
    const std::string dest_path = argv[2];

    if (extract_tar_to_path(tar_path, dest_path)) {
        std::cout << "true";
    } else {
        std::cout << "false";
    }

    return 0;
}
