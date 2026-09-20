# Spending-Intelligence-Credit-Card-Spend-Forecasting-Anomaly-Detection
Full-stack spending intelligence system that ingests real credit card transaction data, trains a per-category time-series forecasting model, and surfaces statistically grounded anomaly alerts with actionable business guidance. The output is a decision-support engine.

## Security summary
This project handles sensitive financial data, so runtime security is important. The repository does not contain committed secrets in the tracked source code we reviewed, and the app uses environment variables for database and cloud credentials instead of hardcoded tokens.

The API has been hardened to reduce exposure:
- Removed the permissive wildcard CORS configuration in `api/main.py`
- Added support for a restricted `CORS_ORIGINS` allowlist
- Added optional bearer-token enforcement through `API_TOKEN`
- Added CSV upload validation for extension, empty payloads, and file-size limits
- Reduced internal error leakage by returning sanitized client-facing messages

Important production guidance:
- Keep all credentials in a runtime secret manager or local `.env` file that is ignored by Git
- Use HTTPS only in production
- Restrict API access behind authentication and authorization
- Keep uploads and anomaly endpoints behind rate limits and request logging
- Use least-privilege database and S3 credentials

## Environment configuration
Copy `.env.example` to `.env` and populate the values for your environment. Do not commit `.env` files.

## Recommended deployment checks
- Validate `CORS_ORIGINS` is limited to trusted domains
- Set `API_TOKEN` in deployment secrets
- Ensure `MAX_CSV_UPLOAD_BYTES` is appropriate for your upload workload
- Keep DB and AWS secrets outside source control
- Review logs for request failures and malformed uploads
