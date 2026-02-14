const { JSDOM } = require('jsdom');
const xpath = require('xpath');
/**
 * Retrieves student information from XML data based on username and password.
 * Returns null if no matching student is found.
 *
 * @param {string} xml_data - The XML string containing student information.
 * @param {string} username - The username to search for.
 * @param {string} password - The password to match.
 * @returns {Object|null} An object containing student's name, age, and citizenship, or null if no student matches.
 */
function getStudentInfo(xml_data, username, password) {
    const dom = new JSDOM(xml_data, { contentType: "application/xml" }); // Set contentType to "application/xml"
    const document = dom.window.document;

    // Create XPath query
    const query = `//student[username="${username}" and password="${password}"]`;

    // Evaluate the XPath expression
    const nodes = xpath.select(query, document);

    // If no matching student is found, return null
    if (nodes.length === 0) {
        return null;
    }

    const student = nodes[0];

    // Return the student information as an object
    return {
        name: xpath.select('string(name)', student),
        age: xpath.select('string(age)', student),
        citizenship: xpath.select('string(citizenship)', student),
    };
}


// Main function to read arguments from the command line and print the result
function main() {
    const args = process.argv.slice(2);
    const xml_data = args[0];
    const username = args[1];
    const password = args[2];

    const studentInfo = getStudentInfo(xml_data, username, password);
    if (!studentInfo) {
        console.log(studentInfo);
        return;
    }
    console.log(studentInfo.name);
    console.log(studentInfo.age);
    console.log(studentInfo.citizenship);
}

main();
