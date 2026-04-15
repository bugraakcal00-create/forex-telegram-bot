from __future__ import annotations

import uvicorn

from app.logging_config import setup_logging
setup_logging()

from app.config import settings


if __name__ == "__main__":
    uvicorn.run(
        "app.web.server:app",
        host=settings.web_host,
        port=settings.web_port,
        reload=False,
    )
