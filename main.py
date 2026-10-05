"""Run one process: pending approval sessions currently live in memory."""

import logging

import uvicorn

from app.api.routes import create_app
from core.config import HOST, PORT

app = create_app()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=HOST, port=PORT)
