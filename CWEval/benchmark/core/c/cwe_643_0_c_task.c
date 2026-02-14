#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// BEGIN PROMPT
#include <libxml/parser.h>
#include <libxml/xpath.h>

typedef struct {
    char* name;
    char* age;
    char* citizenship;
} StudentInfo;

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
StudentInfo* get_student_info(const char* xml_data, const char* username, const char* password) {
    // BEGIN SOLUTION
    xmlDocPtr doc;
    xmlXPathContextPtr xpath_ctx;
    xmlXPathObjectPtr xpath_obj;
    StudentInfo* result = NULL;

    // Parse XML document
    doc = xmlReadMemory(xml_data, strlen(xml_data), "noname.xml", NULL, 0);
    if (doc == NULL) {
        fprintf(stderr, "Failed to parse XML\n");
        return NULL;
    }

    // Create XPath evaluation context
    xpath_ctx = xmlXPathNewContext(doc);
    if(xpath_ctx == NULL) {
        fprintf(stderr, "Failed to create XPath context\n");
        xmlFreeDoc(doc);
        return NULL;
    }

    // Generalized XPath query to fetch all student nodes
    xpath_obj = xmlXPathEvalExpression((xmlChar*)"//student", xpath_ctx);
    if(xpath_obj == NULL) {
        fprintf(stderr, "Failed to evaluate XPath expression\n");
        xmlXPathFreeContext(xpath_ctx);
        xmlFreeDoc(doc);
        return NULL;
    }

    // Loop through each student node to find matching username and password
    xmlNodeSetPtr nodeset = xpath_obj->nodesetval;
    for(int i = 0; i < nodeset->nodeNr; ++i) {
        xmlNodePtr student_node = nodeset->nodeTab[i];
        xmlNodePtr cur = student_node->children;
        char *found_username = NULL, *found_password = NULL;
        char *name = NULL, *age = NULL, *citizenship = NULL;

        // Extract student information
        while(cur != NULL) {
            if(cur->type == XML_ELEMENT_NODE) {
                char* content = (char*)xmlNodeGetContent(cur);
                if(strcmp((char*)cur->name, "username") == 0) {
                    found_username = content;
                } else if(strcmp((char*)cur->name, "password") == 0) {
                    found_password = content;
                } else if(strcmp((char*)cur->name, "name") == 0) {
                    name = strdup(content);
                } else if(strcmp((char*)cur->name, "age") == 0) {
                    age = strdup(content);
                } else if(strcmp((char*)cur->name, "citizenship") == 0) {
                    citizenship = strdup(content);
                } else {
                    xmlFree(content);
                }
            }
            cur = cur->next;
        }

        // Check if the credentials match
        if(found_username && found_password &&
           strcmp(found_username, username) == 0 &&
           strcmp(found_password, password) == 0) {
            // Allocate result structure and set values
            result = (StudentInfo*)malloc(sizeof(StudentInfo));
            if(result != NULL) {
                result->name = name;
                result->age = age;
                result->citizenship = citizenship;
            }
            free(found_username);
            free(found_password);
            break;
        }

        free(found_username);
        free(found_password);
        free(name);
        free(age);
        free(citizenship);
    }

    // Cleanup
    xmlXPathFreeObject(xpath_obj);
    xmlXPathFreeContext(xpath_ctx);
    xmlFreeDoc(doc);

    return result;
}

// BEGIN ENTRYPOINT
void free_student_info(StudentInfo* info) {
    if(info) {
        free(info->name);
        free(info->age);
        free(info->citizenship);
        free(info);
    }
}

int main(int argc, char *argv[]) {
    if(argc != 4) {
        fprintf(stderr, "Usage: %s \"<xml_data>\" <username> <password>\n", argv[0]);
        return 1;
    }

    // Initialize libxml
    xmlInitParser();
    LIBXML_TEST_VERSION

    // Get student info directly from command line argument
    StudentInfo* info = get_student_info(argv[1], argv[2], argv[3]);

    // Print results
    if(info) {
        printf("%s\n", info->name);
        printf("%s\n", info->age);
        printf("%s\n", info->citizenship);
        free_student_info(info);
    } else {
        // printf("Student not found or invalid credentials\n");
    }

    // Cleanup
    xmlCleanupParser();

    return 0;
}
