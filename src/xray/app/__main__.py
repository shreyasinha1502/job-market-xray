"""`python -m xray.app`: serve the dashboard (Render sets PORT)."""

import logging
import os

from xray.app.charts import CSS
from xray.app.dashboard import build
from xray.app.model import ModelUnavailable, predictor
from xray.log import setup_logging

setup_logging(os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger("xray.app")

try:  # load the model once at startup, not on the first request
    predictor()
    log.info("model loaded")
except ModelUnavailable as e:
    log.error(f"classifier disabled: {e}")

build().launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", "7860")),
    css=CSS,
    ssr_mode=False,
    show_error=True,
)
