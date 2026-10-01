"""`python -m xray.app`: serve the site (Render sets PORT)."""

import os

import uvicorn

from xray.log import setup_logging

setup_logging(os.environ.get("LOG_LEVEL", "INFO"))

uvicorn.run(
    "xray.app.server:app",
    host="0.0.0.0",
    port=int(os.environ.get("PORT", "7860")),
    proxy_headers=True,
    forwarded_allow_ips="*",
    log_level="warning",
)
