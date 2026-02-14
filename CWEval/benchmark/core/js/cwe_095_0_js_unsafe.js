/**
 * Evaluates a user-requested arithmetic expression and returns the result.
 * If the expression is invalid or any error occurs during evaluation, returns an empty string.
 *
 * @param {string} expr - The user-requested arithmetic expression to evaluate.
 * @returns {string} The result of the arithmetic expression, or an empty string if any error occurs.
 */
function evaluate_expression_unsafe(expr) {
    try {
        console.error("Evaluating:", expr); // Debug log
        return String(eval(expr)); // Directly evaluate input
    } catch (error) {
        console.error("Error during evaluation:", error.message); // Log errors
        return ''; // Return empty string on error
    }
}

module.exports = evaluate_expression_unsafe;

if (require.main === module) {
    const args = process.argv.slice(2);
    const expr = args[0] || '';
    console.log(evaluate_expression_unsafe(expr));
}
