"""Optional in-process periodic source refresh, gated behind
ENABLE_SCHEDULER=true (api/config.py) so it's never running by default and
never double-triggers alongside an external cron/EventBridge rule hitting
POST /admin/sync-all-sources on the same schedule.

This exists for deployments with no platform-native cron available (or as
the simplest option for a single always-on EC2/Railway/Render instance) --
see GrantSetu_Deployment_Guide.pdf's "Source ingestion" section for the
tradeoffs against platform-native scheduling (Render/Railway Cron Jobs,
AWS EventBridge Scheduler), which is the RECOMMENDED approach for multi-
instance or serverless-style deployments, since an in-process scheduler
enabled on more than one running instance would trigger duplicate syncs.
"""
import logging

from apscheduler.schedulers.background import BackgroundScheduler

from api.config import SYNC_INTERVAL_HOURS

logger = logging.getLogger("grantsetu.scheduler")

_scheduler: BackgroundScheduler | None = None


def _run_scheduled_refresh():
    from api.db.session import SessionLocal
    from api.routers.grant_sources import _execute_refresh_job

    db = SessionLocal()
    try:
        from api.db.models import RefreshJob

        job = RefreshJob(status="running", trigger="scheduled")
        db.add(job)
        db.commit()
        db.refresh(job)
        _execute_refresh_job(db, job.job_id)
    except Exception:
        logger.exception("Scheduled source refresh failed")
    finally:
        db.close()


def start_scheduler() -> BackgroundScheduler | None:
    global _scheduler
    if _scheduler is not None:
        return _scheduler
    _scheduler = BackgroundScheduler(timezone="UTC")
    # APScheduler's "interval" trigger with no explicit start_date already
    # schedules the first run one full interval from now (not immediately
    # at boot) -- fine here, since every source was likely already synced
    # recently by whoever deployed this instance.
    _scheduler.add_job(
        _run_scheduled_refresh,
        "interval",
        hours=SYNC_INTERVAL_HOURS,
        id="refresh_all_sources",
    )
    _scheduler.start()
    logger.info("In-process scheduler started: refresh every %.1f hour(s)", SYNC_INTERVAL_HOURS)
    return _scheduler


def stop_scheduler():
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
