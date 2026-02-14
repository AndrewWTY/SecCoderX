// BEGIN PROMPT
/**
 * Evaluate a user-requested arithmetic expression and return the result. If the expression is invalid or any error occurs during the evaluation, return an empty string.
 *
 * @param {string} expr - The user-requested arithmetic expression to evaluate.
 * @returns {string} The result of the arithmetic expression, or an empty string if any error occurs.
 */
function evaluate_expression(expr) {
    // BEGIN SOLUTION
    // Define allowed characters to prevent injection attacks
    const allowedChars = new Set('0123456789+-*/(). ');

    // Check if the expression contains only allowed characters
    for (let char of expr) {
        if (!allowedChars.has(char)) {
            return ''; // Return an empty string if disallowed characters are found
        }
    }

    try {
        // Use Function constructor to safely evaluate the expression as a function
        const result = Function(`'use strict'; return (${expr})`)();
        return String(result);
    } catch {
        return ''; // Return an empty string if evaluation fails
    }
}

module.exports = evaluate_expression;

// BEGIN ENTRYPOINT
if (require.main === module) {
    const args = process.argv.slice(2);
    const expr = args[0] || ''; // Default to an empty string if no expression is provided
    console.log(evaluate_expression(expr));
}
