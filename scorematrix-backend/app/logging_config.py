import logging


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Library loggers can include request URLs/headers at DEBUG; keep them quiet.
    for noisy in ("httpx", "httpcore", "google_genai", "urllib3", "pypdf"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
