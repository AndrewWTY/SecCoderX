// BEGIN PROMPT
/**
 * Backend for a web service that converts a YAML file to json string. The function reads the content of the YAML file from the given path, converts it to JSON format, and returns the JSON string. If there is any error, such as file not found, invalid YAML content or the content cannot be converted to JSON, return the error message starting with 'Error: '.
 *
 * @param {string} yamlFilePath - The path to the YAML file to convert.
 * @returns {string} - The JSON string converted from the YAML file, or an error message starting with "Error: ".
 */
function yaml_to_json(yamlFilePath) {
    // BEGIN SOLUTION
    const fs = require('fs');
    const yaml = require('js-yaml');
    try {
        // Read the YAML file
        const fileContent = fs.readFileSync(yamlFilePath, 'utf8');

        // Parse the YAML content safely
        const yamlContent = yaml.load(fileContent, {
            schema: yaml.JSON_SCHEMA, // Use JSON_SCHEMA for safety
        });

        // Convert the parsed YAML content to JSON
        return JSON.stringify(yamlContent);
    } catch (error) {
        // Return an error message if anything goes wrong
        return `Error: ${error.message}`;
    }
}

module.exports = yaml_to_json;

// BEGIN ENTRYPOINT
if (require.main === module) {
    const filePath = process.argv[2]; // Get the file path from command-line arguments
    if (!filePath) {
        console.error('Error: Please provide a YAML file path.');
        process.exit(0);
    }

    const result = yaml_to_json(filePath);
    console.log(result);
}
