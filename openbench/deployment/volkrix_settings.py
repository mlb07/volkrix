"""Production overrides for the pinned Volkrix OpenBench instance.

This file is copied into OpenSite/ by scripts/openbench_deploy.py. Secrets are
read only from the process environment; no deployment credential belongs in a
source checkout or generated manifest.
"""

import os

from OpenSite.settings import *  # noqa: F403


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"required deployment variable {name} is unset")
    return value


def enabled(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes"}


SECRET_KEY = required("OPENBENCH_SECRET_KEY")
DEBUG = False
https_enabled = enabled("OPENBENCH_HTTPS", "1")
ALLOWED_HOSTS = [item.strip() for item in required("OPENBENCH_ALLOWED_HOSTS").split(",") if item.strip()]
if not ALLOWED_HOSTS:
    raise RuntimeError("OPENBENCH_ALLOWED_HOSTS must contain at least one host")
CSRF_TRUSTED_ORIGINS = [
    item.strip()
    for item in os.environ.get("OPENBENCH_CSRF_TRUSTED_ORIGINS", "").split(",")
    if item.strip()
]
if https_enabled and not CSRF_TRUSTED_ORIGINS:
    raise RuntimeError("OPENBENCH_CSRF_TRUSTED_ORIGINS is required when HTTPS is enabled")
STATIC_ROOT = os.path.join(BASE_DIR, "StaticRoot")  # noqa: F405

database_backend = os.environ.get("OPENBENCH_DB_ENGINE", "").strip().lower()
if database_backend == "mysql":
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.mysql",
            "NAME": required("OPENBENCH_DB_NAME"),
            "USER": required("OPENBENCH_DB_USER"),
            "PASSWORD": required("OPENBENCH_DB_PASSWORD"),
            "HOST": os.environ.get("OPENBENCH_DB_HOST", "127.0.0.1"),
            "PORT": os.environ.get("OPENBENCH_DB_PORT", "3306"),
            "CONN_MAX_AGE": 60,
            "CONN_HEALTH_CHECKS": True,
        }
    }
elif database_backend == "sqlite" and enabled("OPENBENCH_ALLOW_SQLITE"):
    # This opt-in exists only for an isolated local smoke. The official
    # deployment guidance recommends MySQL for concurrent production workers.
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": os.path.join(BASE_DIR, "db.sqlite3"),  # noqa: F405
        }
    }
else:
    raise RuntimeError(
        "set OPENBENCH_DB_ENGINE=mysql for production, or explicitly opt into "
        "OPENBENCH_DB_ENGINE=sqlite and OPENBENCH_ALLOW_SQLITE=1 for a local smoke"
    )

SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
USE_X_FORWARDED_HOST = True
SECURE_SSL_REDIRECT = https_enabled
SESSION_COOKIE_SECURE = https_enabled
CSRF_COOKIE_SECURE = https_enabled
SECURE_HSTS_SECONDS = 31_536_000 if https_enabled else 0
SECURE_HSTS_INCLUDE_SUBDOMAINS = https_enabled and enabled("OPENBENCH_HSTS_INCLUDE_SUBDOMAINS")
SECURE_HSTS_PRELOAD = https_enabled and enabled("OPENBENCH_HSTS_PRELOAD")
X_FRAME_OPTIONS = "DENY"
