CMD ["bash","-lc","
set -e
mkdir -p /app/.streamlit /root/.streamlit

cat > /app/.streamlit/secrets.toml <<EOF
[auth]
redirect_uri = \"${AUTH_REDIRECT_URI}\"
cookie_secret = \"${AUTH_COOKIE_SECRET}\"
client_id = \"${AUTH_GOOGLE_CLIENT_ID}\"
client_secret = \"${AUTH_GOOGLE_CLIENT_SECRET}\"
server_metadata_url = \"${AUTH_SERVER_METADATA_URL}\"
EOF

cp /app/.streamlit/secrets.toml /root/.streamlit/secrets.toml

streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8080} --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false
"]
