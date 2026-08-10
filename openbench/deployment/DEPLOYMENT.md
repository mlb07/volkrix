# Pinned Volkrix OpenBench instance

`VOLKRIX-DEPLOYMENT.json` identifies and hashes this generated instance. Do not
change its OpenBench or FastChess refs without re-auditing the new upstream
commits and updating Volkrix's `openbench/upstream-lock.json`.

The generator replaces upstream's unsupported Django 4.2.1 pin with the audited
Django 5.2 LTS patch recorded in that lock file. It also applies and hashes two
exact `timezone.now()` compatibility edits required by Django 5.2, prevents the
PGN watcher from running during explicitly marked management commands, and
delays its first production query until application initialization completes.
Do not restore the upstream pin, alter those patches, or upgrade across a
Django feature series without rerunning migrations, checks, and endpoint tests
in a fresh environment.

## Isolated local server smoke

Use Python 3.11 in a dedicated environment. SQLite is deliberately opt-in and
is suitable only for this local smoke:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
export OPENBENCH_SECRET_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(64))')"
export OPENBENCH_ALLOWED_HOSTS=127.0.0.1,localhost
export OPENBENCH_DB_ENGINE=sqlite
export OPENBENCH_ALLOW_SQLITE=1
export OPENBENCH_HTTPS=0
OPENBENCH_DISABLE_WATCHER=1 .venv/bin/python manage.py migrate --settings=OpenSite.volkrix_settings
OPENBENCH_DISABLE_WATCHER=1 .venv/bin/python manage.py check --settings=OpenSite.volkrix_settings
.venv/bin/python manage.py runserver 127.0.0.1:8000 --settings=OpenSite.volkrix_settings
```

The generated secret above is intentionally ephemeral. Do not reuse this smoke
database or secret for deployment.

## Production server

1. Install MySQL, its client development headers, a C compiler, and `pkg-config`
   (for example, Debian/Ubuntu packages `default-libmysqlclient-dev`,
   `build-essential`, and `pkg-config`). Create a least-privilege `openbench`
   database/user and set up encrypted backups with a tested restore procedure.
2. Install the upstream and deployment dependencies into an isolated Python
   3.11 environment:

   ```bash
   uv venv --python 3.11 .venv
   uv pip install --python .venv/bin/python -r requirements.txt -r requirements-deploy.txt
   ```

3. Copy `openbench.env.example` outside the checkout, replace every placeholder,
   restrict it to the service account, and load it into the service environment.
   Keep HSTS subdomains/preload disabled until every subdomain is permanently
   HTTPS-only; enabling preload prematurely is difficult to reverse.
4. Run migrations and deployment checks while the service is offline:

   ```bash
   OPENBENCH_DISABLE_WATCHER=1 .venv/bin/python manage.py migrate --settings=OpenSite.volkrix_settings
   OPENBENCH_DISABLE_WATCHER=1 .venv/bin/python manage.py collectstatic --noinput --settings=OpenSite.volkrix_settings
   OPENBENCH_DISABLE_WATCHER=1 .venv/bin/python manage.py check --deploy --settings=OpenSite.volkrix_settings
   ```

   Django will intentionally retain its HSTS subdomain/preload warnings while
   those two opt-ins are disabled. Resolve them only after the entire domain is
   eligible; every other deployment warning must be resolved before launch.

5. Run Gunicorn only on loopback and put it behind an HTTPS Nginx reverse proxy.
   The official OpenBench guidance uses three workers and a 250 MB upload limit:

   ```bash
   DJANGO_SETTINGS_MODULE=OpenSite.volkrix_settings \
     .venv/bin/gunicorn OpenSite.wsgi:application \
       --bind 127.0.0.1:8000 --workers 3 --graceful-timeout 300 --timeout 120
   ```

6. Configure Nginx to serve `StaticRoot/` at `/static/`, proxy all application
   endpoints to `127.0.0.1:8000`, preserve `Host` and `X-Forwarded-*` headers,
   set `client_max_body_size 250M`, and obtain/renew a trusted TLS certificate.
7. Always stop OpenBench/Gunicorn with SIGTERM and wait for graceful exit. SIGKILL
   can leave PGN archives or watcher state incomplete.
8. Create the Django superuser, then the ordinary OpenBench account. Enable its
   profile and approver permissions before creating workloads or managing nets.
9. Upload the SHA-256-verified production NNUE through Network administration.
   Keep the network source/license record and select the same network for both
   sides of ordinary change tests.

## Worker

Workers need Python `requests`, Make, a C++ compiler for FastChess, Git, and
Cargo 1.85 or newer. Clone the exact OpenBench commit from
`VOLKRIX-DEPLOYMENT.json`, install `requests` in a dedicated environment, and
provide credentials only at runtime:

```bash
export OPENBENCH_USERNAME=REPLACE
export OPENBENCH_PASSWORD=REPLACE
export OPENBENCH_SERVER=https://openbench.example.com
python3 Client/client.py --no-client-downloads \
  -T REPLACE_WITH_STABLE_PHYSICAL_CORE_COUNT -N 1 -I worker-name --focus Volkrix
```

Use the physical-core count for `-T`, the actual socket count for `-N`, and
reduce `-T` if measured memory, thermals, or clock stability are poor. Keep the
worker clock synchronized. Never commit credential files or populated service
environment files.

On heterogeneous CPUs, do not blindly count performance and efficiency cores as
equivalent. Start a reference worker with only the performance-core count and
confirm that OpenBench's simultaneous bench sets have stable node counts and
speeds before trusting time-scaled results.

Before accepting results, run a no-change STC workload and require a neutral
result, identical bench node counts, and zero crashes, stalls, time forfeits, or
illegal moves. Then validate an intentionally harmless source change end to end.
