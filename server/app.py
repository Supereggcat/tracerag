"""Convenience module for `uvicorn server.app:app --reload`."""

from tracerag.webapp import create_app

app = create_app()
