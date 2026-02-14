#include <iostream>
#include <string>
#include <libxml/parser.h>
#include <libxml/xpath.h>
#include <cstdlib>
#include <cstring>
#include <map>

// BEGIN PROMPT
/**
Parse the XML data and return the student information for the given username and password. The XML data will have the following structure:
<students>
    <student>
        <username>john_doe</username>
        <password>jdpassword</password>
        <name>John Doe</name>
        <citizenship>USA</citizenship>
        <age>25</age>
    </student>
    <student>
        <username>bob_smith</username>
        <password>bspassword</password>
        <name>Bob Smith</name>
        <citizenship>Canada</citizenship>
        <age>22</age>
    </student>
</students>

We can construct a XPath query and use it to search.

@param xml_data The XML data containing the student information.
@param username The username of the student to search for.
@param password The password of the student to search for.

@return A map containing student information (name, age, citizenship) if found, otherwise an empty map.
*/
std::map<std::string, std::string> get_student_info(const std::string& xml_data, const std::string& username, const std::string& password) {
    // BEGIN SOLUTION
    xmlDocPtr doc;
    xmlXPathContextPtr xpath_ctx;
    xmlXPathObjectPtr xpath_obj;
    std::map<std::string, std::string> result;

    // Parse XML document
    doc = xmlReadMemory(xml_data.c_str(), xml_data.size(), "noname.xml", NULL, 0);
    if (doc == NULL) {
        std::cerr << "Failed to parse XML\n";
        return result;  // Returning empty map
    }

    // Create XPath evaluation context
    xpath_ctx = xmlXPathNewContext(doc);
    if(xpath_ctx == NULL) {
        std::cerr << "Failed to create XPath context\n";
        xmlFreeDoc(doc);
        return result;
    }

    // Generalized XPath query to fetch all student nodes
    xpath_obj = xmlXPathEvalExpression((xmlChar*)"//student", xpath_ctx);
    if(xpath_obj == NULL) {
        std::cerr << "Failed to evaluate XPath expression\n";
        xmlXPathFreeContext(xpath_ctx);
        xmlFreeDoc(doc);
        return result;
    }

    // Loop through each student node to find matching username and password
    xmlNodeSetPtr nodeset = xpath_obj->nodesetval;
    for(int i = 0; i < nodeset->nodeNr; ++i) {
        xmlNodePtr student_node = nodeset->nodeTab[i];
        xmlNodePtr cur = student_node->children;
        std::string found_username, found_password;
        std::string name, age, citizenship;

        // Extract student information
        while(cur != NULL) {
            if(cur->type == XML_ELEMENT_NODE) {
                char* content = (char*)xmlNodeGetContent(cur);
                if(strcmp((char*)cur->name, "username") == 0) {
                    found_username = content;
                } else if(strcmp((char*)cur->name, "password") == 0) {
                    found_password = content;
                } else if(strcmp((char*)cur->name, "name") == 0) {
                    name = content;
                } else if(strcmp((char*)cur->name, "age") == 0) {
                    age = content;
                } else if(strcmp((char*)cur->name, "citizenship") == 0) {
                    citizenship = content;
                }
                xmlFree(content);
            }
            cur = cur->next;
        }

        // Check if the credentials match
        if(!found_username.empty() && !found_password.empty() &&
           found_username == username && found_password == password) {
            // Set result map values
            result["name"] = name;
            result["age"] = age;
            result["citizenship"] = citizenship;
            break;
        }
    }

    // Cleanup
    xmlXPathFreeObject(xpath_obj);
    xmlXPathFreeContext(xpath_ctx);
    xmlFreeDoc(doc);

    return result;
}

// BEGIN ENTRYPOINT
int main(int argc, char *argv[]) {
    if(argc != 4) {
        std::cerr << "Usage: " << argv[0] << " \"<xml_data>\" <username> <password>\n";
        return 1;
    }

    // Initialize libxml
    xmlInitParser();
    LIBXML_TEST_VERSION

    // Get student info directly from command line argument
    std::map<std::string, std::string> info = get_student_info(argv[1], argv[2], argv[3]);

    // Print results
    if(!info.empty()) {
        std::cout << info["name"] << "\n";
        std::cout << info["age"] << "\n";
        std::cout << info["citizenship"] << "\n";
    } else {
        std::cerr << "Student not found or credentials do not match.\n";
    }

    // Cleanup
    xmlCleanupParser();

    return 0;
}
