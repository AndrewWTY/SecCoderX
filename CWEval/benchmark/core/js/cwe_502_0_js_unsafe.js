const fs = require('fs');
const yaml = require('js-yaml');

function yaml_to_json_unsafe(yamlFilePath) {
    try {
        // Read the YAML file
        const fileContent = fs.readFileSync(yamlFilePath, 'utf8');

        // Parse the YAML content without restricting schema or validations
        const yamlContent = yaml.load(fileContent, {
            schema: yaml.DEFAULT_SCHEMA, // Allows custom tags and potentially dangerous constructs
        });

        // Convert the parsed YAML content to JSON
        return JSON.stringify(yamlContent);
    } catch (error) {
        // Return an error message if anything goes wrong
        return `Error: ${error.message}`;
    }
}

module.exports = yaml_to_json_unsafe;

// Example usage (CLI or script)
if (require.main === module) {
    const filePath = process.argv[2]; // Get the file path from command-line arguments
    if (!filePath) {
        console.error('Error: Please provide a YAML file path.');
        process.exit(0);
    }

    const result = yaml_to_json_unsafe(filePath);
    console.log(result);
}
