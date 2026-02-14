# API Key Configuration

This folder contains API key configuration files for LLM providers.

## Setup

1. **Anthropic Claude**:
   - Open `anthropic_api_key.txt`
   - Replace `your-anthropic-api-key-here` with your actual API key
   - Save the file

2. **OpenAI GPT**:
   - Open `openai_api_key.txt`
   - Replace `your-openai-api-key-here` with your actual API key
   - Save the file

3. **Google Gemini**:
   - Open `gemini_api_key.txt`
   - Replace `your-google-api-key-here` with your actual API key
   - Save the file

## File Format

Each file should contain only the API key on a single line with no extra spaces or newlines.

Example:
```
sk-ant-api03-1234567890abcdef
```

## Security

- These files are already added to `.gitignore` to prevent accidental commits
- Never commit your actual API keys to version control
- Keep these files private and secure

## Alternative: Environment Variables

If you prefer, you can still use environment variables instead:

```bash
export ANTHROPIC_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"
export GOOGLE_API_KEY="your-key-here"
```

The system will first check for the text files in this folder, then fall back to environment variables if the files don't exist.
