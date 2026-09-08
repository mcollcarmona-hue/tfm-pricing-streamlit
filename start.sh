#!/bin/bash
pip install -r requirements.txt
npm install -g localtunnel
echo "=== TU CONTRASEÑA DE LOCALTUNNEL (ENDPOINT IP) ES: ==="
curl -s ipv4.icanhazip.com
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false & npx localtunnel --port 8501 --subdomain tfm-pricing-mcoll
