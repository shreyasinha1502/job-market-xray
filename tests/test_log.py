import json
import logging

from xray.log import JsonFormatter, kv, redact, register_secret


def test_redacts_credential_query_params():
    url = "https://api.adzuna.com/v1/api/jobs/in/search/1?app_id=abc123&app_key=def456&what=x"
    out = redact(url)
    assert "abc123" not in out and "def456" not in out
    assert "app_id=***" in out and "what=x" in out


def test_redacts_registered_secret_anywhere():
    register_secret("s3cr3t-value-for-test")
    assert "s3cr3t" not in redact("error: bad key s3cr3t-value-for-test in header")


def test_json_formatter_emits_structured_fields():
    rec = logging.LogRecord("t", logging.INFO, __file__, 1, "page fetched", None, None)
    rec.fields = kv(page=2, n=50)["fields"]
    payload = json.loads(JsonFormatter().format(rec))
    assert payload["msg"] == "page fetched" and payload["page"] == 2 and payload["n"] == 50
