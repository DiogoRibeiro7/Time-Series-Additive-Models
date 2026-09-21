# Security Policy

## Reporting a vulnerability

Please do not open a public issue for credentials, secrets, or exploitable vulnerabilities.

Use GitHub's private vulnerability reporting feature when available.

## Credentials

API keys and tokens must never be committed to the repository. Use environment variables or secret stores for local and CI configuration.

Any credential accidentally committed to Git history must be considered compromised and rotated immediately.
