package main

import (
	"bytes"
	"fmt"
	"os"
	"strings"

	"github.com/ChrisTrenkamp/goxpath"
	"github.com/ChrisTrenkamp/goxpath/tree"
	"github.com/ChrisTrenkamp/goxpath/tree/xmltree"
)

// BEGIN PROMPT
/*
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

Use these libs:
"github.com/ChrisTrenkamp/goxpath"
"github.com/ChrisTrenkamp/goxpath/tree"
"github.com/ChrisTrenkamp/goxpath/tree/xmltree"

We can construct a XPath query and then execute the query to search for the student information.

@param xmlData The XML data containing the student information.
@param username The username of the student to search for.
@param password The password of the student to search for.

@return A tuple (name, age, citizenship) if the student is found. Otherwise, returns empty strings.
*/
func getStudentInfo(xmlData, username, password string) (string, string, string) {
	// BEGIN SOLUTION
	// Parse XML document
	xTree, err := xmltree.ParseXML(bytes.NewBufferString(xmlData))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to parse XML: %v\n", err)
		return "", "", ""
	}

	// Prepare XPath query with variables to prevent injection
	xp := goxpath.MustParse("//student[username=$user and password=$pass]")

	// Evaluate XPath expression with registered variables
	res, err := xp.ExecNode(xTree, func(o *goxpath.Opts) {
		o.Vars["user"] = tree.String(username)
		o.Vars["pass"] = tree.String(password)
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to evaluate XPath expression: %v\n", err)
		return "", "", ""
	}

	// Check if we found a matching student
	if len(res) > 0 {
		if studentNode, ok := res[0].(tree.Elem); ok {
			var name, age, citizenship string

			// Helper function to get child element content
			getChildContent := func(nodeName string) string {
				childXPath := fmt.Sprintf("./%s", nodeName)
				childXp := goxpath.MustParse(childXPath)
				if children, err := childXp.ExecNode(studentNode); err == nil && len(children) > 0 {
					return strings.TrimSpace(children[0].ResValue())
				}
				return ""
			}

			// Extract student information
			name = getChildContent("name")
			age = getChildContent("age")
			citizenship = getChildContent("citizenship")

			return name, age, citizenship
		}
	}

	return "", "", ""
}

// BEGIN ENTRYPOINT
func main() {
	// Check command line arguments
	if len(os.Args) != 4 {
		fmt.Fprintf(os.Stderr, "Usage: %s \"<xml_data>\" <username> <password>\n", os.Args[0])
		os.Exit(1)
	}

	// Get student info directly from command line arguments
	name, age, citizenship := getStudentInfo(os.Args[1], os.Args[2], os.Args[3])

	// Print results
	if name != "" || age != "" || citizenship != "" {
		fmt.Println(name)
		fmt.Println(age)
		fmt.Println(citizenship)
	}
}
