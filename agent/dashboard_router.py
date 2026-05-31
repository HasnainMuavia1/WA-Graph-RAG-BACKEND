"""
Dashboard statistics endpoints - /api/v1/dashboard/stats
All routes require an authenticated active dashboard user.
Features 60-second Redis caching.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException

from .auth_deps import get_current_active_user
from .redis_utils import get_redis_client

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])

_auth = get_current_active_user
CACHE_KEY = "uchenab:dashboard:stats"
CACHE_TTL = 60  # 60 seconds cache


def _get_supabase_client():
    from . import db_utils

    if db_utils._client is None:
        raise RuntimeError("Supabase client not initialized")
    return db_utils._client


@router.get("/stats")
async def get_dashboard_stats_endpoint(current_user=Depends(_auth)):
    """Fetch total conversation counts and 7-day message trends.

    Results are cached in Redis for 60 seconds to protect the PostgreSQL/Supabase database.
    """
    # 1. Try serving from Redis cache
    try:
        redis_client = get_redis_client()
        if redis_client:
            cached_data = redis_client.get(CACHE_KEY)
            if cached_data:
                logger.info("Serving dashboard stats from Redis cache")
                return json.loads(cached_data.decode("utf-8"))
    except Exception as cache_err:
        logger.warning("Failed to fetch dashboard stats from Redis: %s", cache_err)

    # 2. Cache miss: Fetch from Supabase
    try:
        client = _get_supabase_client()

        # A. Fetch total chats
        chats_res = (
            await client.table("wa_conversations").select("id", count="exact").execute()
        )
        total_chats = (
            chats_res.count
            if chats_res.count is not None
            else len(chats_res.data or [])
        )

        # B. Fetch agent replies
        agent_res = (
            await client.table("wa_messages")
            .select("id", count="exact")
            .eq("sender", "agent")
            .execute()
        )
        agent_messages = (
            agent_res.count
            if agent_res.count is not None
            else len(agent_res.data or [])
        )

        # C. Fetch admin replies
        admin_res = (
            await client.table("wa_messages")
            .select("id", count="exact")
            .eq("sender", "admin")
            .execute()
        )
        admin_messages = (
            admin_res.count
            if admin_res.count is not None
            else len(admin_res.data or [])
        )

        # D. Fetch user messages received
        user_res = (
            await client.table("wa_messages")
            .select("id", count="exact")
            .eq("sender", "user")
            .execute()
        )
        user_messages = (
            user_res.count if user_res.count is not None else len(user_res.data or [])
        )

        # E. Calculate 7-day trend
        now_utc = datetime.now(timezone.utc)
        start_date = (now_utc - timedelta(days=6)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        start_date_str = start_date.isoformat()

        # Fetch messages in the last 7 days
        msg_trend_res = (
            await client.table("wa_messages")
            .select("sender, created_at")
            .gte("created_at", start_date_str)
            .execute()
        )
        messages_in_period = msg_trend_res.data or []

        # Fetch conversations started in the last 7 days
        conv_trend_res = (
            await client.table("wa_conversations")
            .select("created_at")
            .gte("created_at", start_date_str)
            .execute()
        )
        conversations_in_period = conv_trend_res.data or []

        # Build dates structure
        trends = []
        for i in range(6, -1, -1):
            day = (now_utc - timedelta(days=i)).strftime("%Y-%m-%d")
            trends.append(
                {
                    "date": day,
                    "chats_started": 0,
                    "user_messages": 0,
                    "agent_messages": 0,
                    "admin_messages": 0,
                }
            )

        # Aggregate messages by date
        for msg in messages_in_period:
            created_at_str = msg.get("created_at")
            if not created_at_str:
                continue
            date_str = created_at_str[:10]  # Get YYYY-MM-DD
            for t in trends:
                if t["date"] == date_str:
                    sender = msg.get("sender")
                    if sender == "user":
                        t["user_messages"] += 1
                    elif sender == "agent":
                        t["agent_messages"] += 1
                    elif sender == "admin":
                        t["admin_messages"] += 1
                    break

        # Aggregate started conversations by date
        for conv in conversations_in_period:
            created_at_str = conv.get("created_at")
            if not created_at_str:
                continue
            date_str = created_at_str[:10]  # Get YYYY-MM-DD
            for t in trends:
                if t["date"] == date_str:
                    t["chats_started"] += 1
                    break

        response_data = {
            "totals": {
                "total_chats": total_chats,
                "user_messages": user_messages,
                "agent_messages": agent_messages,
                "admin_messages": admin_messages,
            },
            "trends": trends,
            "cached": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # 3. Save to Redis cache for subsequent loads
        try:
            redis_client = get_redis_client()
            if redis_client:
                # We flag as cached: True in the cached version so we can track it
                cached_version = response_data.copy()
                cached_version["cached"] = True
                redis_client.setex(CACHE_KEY, CACHE_TTL, json.dumps(cached_version))
                logger.info("Saved dashboard stats to Redis cache (TTL=%d)", CACHE_TTL)
        except Exception as cache_err:
            logger.warning("Failed to save dashboard stats to Redis: %s", cache_err)

        return response_data

    except Exception as db_err:
        logger.error("Failed to query dashboard statistics: %s", db_err)
        raise HTTPException(status_code=500, detail=str(db_err))
