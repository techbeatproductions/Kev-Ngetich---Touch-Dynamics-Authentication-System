# config.py

CELERY_BROKER_URL = 'redis://localhost:6379/0'  # URL of the Redis broker
CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'  # URL of the Redis result backend
