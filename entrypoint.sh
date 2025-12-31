#!/bin/sh
set -eu

mkdir -p /root/.streamlit

# Write Streamlit secrets from Cloud Run env vars
cat > /root/.streamlit/secrets.toml <<EOF
GEMINI_API_KEY = "${GEMINI_API_KEY}"

ALLOWED_DOMAINS = "${ALLOWED_DOMAINS}"
ALLOWED_EMAILS = "${ALLOWED_EMAILS}"

COOKIE_SECRET = "${AUTH_COOKIE_SECRET}"

[auth.google]
client_id = "${AUTH_GOOGLE_CLIENT_ID}"
client_secret = "${AUTH_GOOGLE_CLIENT_SECRET}"
EOF

exec streamlit run app.py --server.address=0.0.0.0 --server.port="${PORT:-8080}"
