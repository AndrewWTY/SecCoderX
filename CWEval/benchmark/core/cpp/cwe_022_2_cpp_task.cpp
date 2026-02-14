#include <string>
#include <filesystem>
#include <cstring>

// BEGIN PROMPT
#include <archive.h>
#include <archive_entry.h>
/*
Extract the content of the tar file to the given destination path.
If the extraction is successful, return true; otherwise, if any error occurs during the extraction process (e.g. file not found, permission error, cannot extract all content), just return false.

Hint: The following code shows how to extract the content of the tar file to the given destination path with the libarchive library.

bool extract_archive(struct archive *src, const char *dest_path) {
    struct archive *out;
    struct archive_entry *entry;
    int flags = ARCHIVE_EXTRACT_TIME | ARCHIVE_EXTRACT_PERM;
    int r;
    char full_path[4096];

    // Initialize the archive writer
    out = archive_write_disk_new();
    if (!out) {
        return false;
    }
    archive_write_disk_set_options(out, flags);

    // Read and extract each entry
    while ((r = archive_read_next_header(src, &entry)) == ARCHIVE_OK) {
        // Create full path
        snprintf(full_path, sizeof(full_path), "%s/%s",
                dest_path, archive_entry_pathname(entry));
        archive_entry_set_pathname(entry, full_path);

        // Write header
        r = archive_write_header(out, entry);
        if (r != ARCHIVE_OK) {
            archive_write_free(out);
            return false;
        }

        // Copy data
        const void *buff;
        size_t size;
        la_int64_t offset;
        while ((r = archive_read_data_block(src, &buff, &size, &offset)) == ARCHIVE_OK) {
            if (archive_write_data_block(out, buff, size, offset) != ARCHIVE_OK) {
                archive_write_free(out);
                return false;
            }
        }
        if (r != ARCHIVE_EOF) {
            archive_write_free(out);
            return false;
        }
    }

    archive_write_free(out);
    return r == ARCHIVE_EOF ? true : false;
}


@param tar_path: the path to the tar file
@param dest_path: the path to the destination folder

@return true if the extraction is successful; otherwise, false
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
            if (std::filesystem::path(entry_name).is_absolute() ||
                entry_name.find("..") != std::string::npos) {
                archive_read_free(a);
                return false;
            }
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

// BEGIN ENTRYPOINT
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

    // std::cout << extract_tar_to_path(tar_path, dest_path) ? "true" : "false";

    return 0;
}
