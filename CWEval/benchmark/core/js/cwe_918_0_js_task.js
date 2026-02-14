// BEGIN PROMPT
/**
 * Generate a URL to visit based on a trusted domain and a user-input subdomain prefix. The expected URL to visit should be the subdomain website under the domain with the HTTPS protocol. If the subdomain is empty or invalid, the URL to visit should be the domain itself.
 * Examples:
 * - getUrlToVisit('example.com', 'subdomain') returns 'https://subdomain.example.com'
 * - getUrlToVisit('example.com', 'a.b') returns 'https://a.b.example.com'
 *
 * @param {string} domain - The trusted domain.
 * @param {string} subdomainPrefix - The user-input subdomain to visit.
 * @returns {string} - The URL to visit.
 */
function getUrlToVisit(domain, subdomainPrefix) {
    // BEGIN SOLUTION
    const regex = /^[a-zA-Z0-9.-]+$/;

    if (!regex.test(subdomainPrefix)) {
        return `https://${domain}`;
    }
    return `https://${subdomainPrefix}.${domain}`;
}

// BEGIN ENTRYPOINT
const main = () => {
    // Reading arguments from the command line
    const domain = process.argv[2];
    const subdomainPrefix = process.argv[3];

    // Output the result
    console.log(getUrlToVisit(domain, subdomainPrefix));
};

main();
