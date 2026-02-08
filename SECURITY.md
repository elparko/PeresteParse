# Security Guidelines

## API Key Storage

- API keys are stored in plain text in `~/.pereste/config.json`
- **Important**: Treat this file as sensitive. Do not share it.
- Set appropriate file permissions: `chmod 600 ~/.pereste/config.json`
- Consider using macOS Keychain for production deployments

## User Data

- All study entries are stored locally in `~/.pereste/entries.json`
- Audio transcriptions are processed locally or sent to Anthropic API (if configured)
- No data is collected or transmitted by the application itself
- Debug logs are stored in `~/.pereste/debug.log` with automatic rotation (10MB max, 3 backups)

## Security Features

Pereste Parse implements several security measures:

- **CORS Restriction**: API access is limited to localhost only
- **Path Traversal Protection**: Model file operations validate filenames
- **Input Sanitization**: Transcriptions are sanitized before LLM processing
- **CSV Import Validation**: File size limits (50MB), encoding validation, and field whitelisting
- **Secure Headers**: X-Frame-Options, Content-Security-Policy, and other security headers
- **Log Rotation**: Prevents unbounded log growth
- **Config Validation**: Input validation for all configuration parameters

## Privacy Considerations

### Local Mode (Default)
- All processing happens on your Mac
- No data leaves your computer
- Models are downloaded once from HuggingFace and cached locally

### Cloud Mode (Optional)
- Transcriptions are sent to Anthropic's API when using cloud parsing
- Review Anthropic's privacy policy: https://www.anthropic.com/privacy
- API keys are stored locally and never transmitted except to authenticate with Anthropic

## Reporting Vulnerabilities

If you discover a security vulnerability in Pereste Parse, please report it by:

1. Opening a [GitHub Issue](https://github.com/elparko/PeresteParse/issues) with the "security" label
2. Emailing the details directly (check GitHub profile for contact info)

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if you have one)

We will respond to security issues as quickly as possible and credit reporters in the fix announcement.

## Security Best Practices

When using Pereste Parse:

1. **Keep the app updated**: Check for updates regularly
2. **Protect your API keys**: Never commit config files to version control
3. **Review imported data**: Only import CSV files from trusted sources
4. **Use local mode for sensitive data**: If privacy is critical, use local LLM processing
5. **Set file permissions**: Restrict access to `~/.pereste/` directory
   ```bash
   chmod 700 ~/.pereste
   chmod 600 ~/.pereste/config.json
   chmod 600 ~/.pereste/entries.json
   ```

## Known Limitations

- The app is **not code-signed**: macOS will show security warnings on first launch
- Local LLMs may have different security characteristics than cloud models
- Prompt injection protection is basic - do not paste untrusted content into transcription fields
- The app runs a local web server on port 5111 (bound to localhost only)

## Third-Party Dependencies

Pereste Parse relies on several third-party libraries. We periodically audit dependencies for security vulnerabilities. To check for known vulnerabilities:

```bash
uv pip install pip-audit
uv run pip-audit
```

Major dependencies:
- **Flask**: Web framework (CORS restricted to localhost)
- **llama-cpp-python**: Local LLM inference
- **anthropic**: Cloud API client
- **parakeet-mlx**: Speech-to-text
- **genanki**: Anki package generation

## License

Pereste Parse is open source under the MIT License. See [LICENSE](LICENSE) for details.
