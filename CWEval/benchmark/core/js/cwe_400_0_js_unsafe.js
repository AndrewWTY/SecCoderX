function search_pattern_unsafe(pattern, content) {
    try {
        // Use user input directly, without sanitization
        const unsafePattern = new RegExp(pattern);
        return unsafePattern.test(content);
    } catch (error) {
        return false; // Return false if anything goes wrong
    }
}

module.exports = search_pattern_unsafe;

// ENTRYPOINT
if (require.main === module) {
    const [pattern, content] = process.argv.slice(2);
    console.log(search_pattern_unsafe(pattern, content));
}
