cat > /root/.streamlit/secrets.toml <<EOF
GEMINI_API_KEY = "${GEMINI_API_KEY}"

ALLOWED_DOMAINS = "${ALLOWED_DOMAINS}"
ALLOWED_EMAILS = "${ALLOWED_EMAILS}"

COOKIE_SECRET = "${AUTH_COOKIE_SECRET}"

[auth]
redirect_uri = "https://gov-plain-language-657594795860.northamerica-northeast2.run.app/oauth2callback"

[auth.google]
client_id = "${AUTH_GOOGLE_CLIENT_ID}"
client_secret = "${AUTH_GOOGLE_CLIENT_SECRET}"
EOF
