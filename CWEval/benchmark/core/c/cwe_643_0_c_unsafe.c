#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libxml/parser.h>
#include <libxml/xpath.h>

typedef struct {
    char* name;
    char* age;
    char* citizenship;
} StudentInfo;

StudentInfo* get_student_info(const char* xml_data, const char* username, const char* password) {
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

    // Create xpath evaluation context
    xpath_ctx = xmlXPathNewContext(doc);
    if(xpath_ctx == NULL) {
        fprintf(stderr, "Failed to create XPath context\n");
        xmlFreeDoc(doc);
        return NULL;
    }

    // Prepare XPath query
    char xpath_query[256];
    snprintf(xpath_query, sizeof(xpath_query),
             "//student[username='%s' and password='%s']",
             username, password);

    // Evaluate XPath expression
    xpath_obj = xmlXPathEvalExpression((xmlChar*)xpath_query, xpath_ctx);
    if(xpath_obj == NULL) {
        fprintf(stderr, "Failed to evaluate XPath expression\n");
        xmlXPathFreeContext(xpath_ctx);
        xmlFreeDoc(doc);
        return NULL;
    }

    // Check if we found a matching student
    if(xpath_obj->nodesetval && xpath_obj->nodesetval->nodeNr > 0) {
        xmlNodePtr student_node = xpath_obj->nodesetval->nodeTab[0];
        xmlNodePtr cur = student_node->children;

        // Allocate result structure
        result = (StudentInfo*)malloc(sizeof(StudentInfo));
        if(result == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            xmlXPathFreeObject(xpath_obj);
            xmlXPathFreeContext(xpath_ctx);
            xmlFreeDoc(doc);
            return NULL;
        }

        // Initialize result fields to NULL
        result->name = NULL;
        result->age = NULL;
        result->citizenship = NULL;

        // Extract student information
        while(cur != NULL) {
            if(cur->type == XML_ELEMENT_NODE) {
                char* content = (char*)xmlNodeGetContent(cur);
                if(strcmp((char*)cur->name, "name") == 0) {
                    result->name = strdup(content);
                } else if(strcmp((char*)cur->name, "age") == 0) {
                    result->age = strdup(content);
                } else if(strcmp((char*)cur->name, "citizenship") == 0) {
                    result->citizenship = strdup(content);
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
