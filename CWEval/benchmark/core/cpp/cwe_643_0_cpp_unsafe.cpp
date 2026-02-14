#include <iostream>
#include <string>
#include <libxml/parser.h>
#include <libxml/xpath.h>
#include <cstring>

struct StudentInfo {
    std::string name;
    std::string age;
    std::string citizenship;
};

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

@return A pointer to a StudentInfo structure if the student is found. Otherwise, returns NULL.
*/
StudentInfo* get_student_info(const std::string& xml_data, const std::string& username, const std::string& password) {
    xmlDocPtr doc;
    xmlXPathContextPtr xpath_ctx;
    xmlXPathObjectPtr xpath_obj;
    StudentInfo* result = nullptr;

    // Parse XML document
    doc = xmlReadMemory(xml_data.c_str(), xml_data.size(), "noname.xml", NULL, 0);
    if (doc == NULL) {
        std::cerr << "Failed to parse XML\n";
        return nullptr;
    }

    // Create xpath evaluation context
    xpath_ctx = xmlXPathNewContext(doc);
    if(xpath_ctx == NULL) {
        std::cerr << "Failed to create XPath context\n";
        xmlFreeDoc(doc);
        return nullptr;
    }

    // Prepare XPath query
    std::string xpath_query = "//student[username='" + username + "' and password='" + password + "']";

    // Evaluate XPath expression
    xpath_obj = xmlXPathEvalExpression((xmlChar*)xpath_query.c_str(), xpath_ctx);
    if(xpath_obj == NULL) {
        std::cerr << "Failed to evaluate XPath expression\n";
        xmlXPathFreeContext(xpath_ctx);
        xmlFreeDoc(doc);
        return nullptr;
    }

    // Check if we found a matching student
    if(xpath_obj->nodesetval && xpath_obj->nodesetval->nodeNr > 0) {
        xmlNodePtr student_node = xpath_obj->nodesetval->nodeTab[0];
        xmlNodePtr cur = student_node->children;

        // Allocate result structure
        result = new StudentInfo();
        if(result == nullptr) {
            std::cerr << "Memory allocation failed\n";
            xmlXPathFreeObject(xpath_obj);
            xmlXPathFreeContext(xpath_ctx);
            xmlFreeDoc(doc);
            return nullptr;
        }

        // Initialize result fields to empty
        result->name = "";
        result->age = "";
        result->citizenship = "";

        // Extract student information
        while(cur != nullptr) {
            if(cur->type == XML_ELEMENT_NODE) {
                char* content = (char*)xmlNodeGetContent(cur);
                if(strcmp((char*)cur->name, "name") == 0) {
                    result->name = content;
                } else if(strcmp((char*)cur->name, "age") == 0) {
                    result->age = content;
                } else if(strcmp((char*)cur->name, "citizenship") == 0) {
                    result->citizenship = content;
                }
                xmlFree(content);
            }
            cur = cur->next;
        }
    }

    // Cleanup
    xmlXPathFreeObject(xpath_obj);
    xmlXPathFreeContext(xpath_ctx);
    xmlFreeDoc(doc);

    return result;
}

/**
 * Free the memory allocated for student info.
 */
void free_student_info(StudentInfo* info) {
    delete info;
}

int main(int argc, char *argv[]) {
    if(argc != 4) {
        std::cerr << "Usage: " << argv[0] << " \"<xml_data>\" <username> <password>\n";
        return 1;
    }

    // Initialize libxml
    xmlInitParser();
    LIBXML_TEST_VERSION

    // Get student info directly from command line argument
    std::string xml_data = argv[1];
    std::string username = argv[2];
    std::string password = argv[3];

    StudentInfo* info = get_student_info(xml_data, username, password);

    // Print results
    if(info) {
        std::cout << info->name << std::endl;
        std::cout << info->age << std::endl;
        std::cout << info->citizenship << std::endl;
        free_student_info(info);
    } else {
        // std::cout << "Student not found or invalid credentials\n";
    }

    // Cleanup
    xmlCleanupParser();

    return 0;
}
