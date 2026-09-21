web: gunicorn api.main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-8000} --workers ${WEB_CONCURRENCY:-2} --timeout 120 --access-logfile - --error-logfile -
release: python -c "from api.db.migrations import create_all_tables; create_all_tables(); print('Migrations applied.')"
