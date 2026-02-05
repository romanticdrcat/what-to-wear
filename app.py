# app.py
from __future__ import annotations

import json
import sqlite3
import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from streamlit_geolocation import streamlit_geolocation

from urllib.parse import urlencode

# [추가] 기상청 호출용
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from zoneinfo import ZoneInfo
import math

# OpenAI (Responses API)
# pip install openai
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


APP_TITLE = "오늘 뭐 입지"
DB_PATH = "ready_to_wear.sqlite"

# [추가] 기상청 단기예보(동네예보) 엔드포인트
KMA_VILAGE_BASE_URL = "https://kma-proxy-worker.pages.dev/api/kma"


# =========================
# [추가] KMA GRID CONVERSION + VILAGE FCST FETCH
# =========================
def latlon_to_grid(lat: float, lon: float) -> Tuple[int, int]:
    """
    위경도 -> 기상청 격자(nx, ny) 변환 (LCC DFS)
    널리 쓰이는 표준 상수(RE, GRID, SLAT1/2, OLON/OLAT, XO/YO) 기반. :contentReference[oaicite:3]{index=3}
    """
    RE = 6371.00877  # km
    GRID = 5.0       # km
    SLAT1 = 30.0
    SLAT2 = 60.0
    OLON = 126.0
    OLAT = 38.0
    XO = 43
    YO = 136

    DEGRAD = math.pi / 180.0

    re = RE / GRID
    slat1 = SLAT1 * DEGRAD
    slat2 = SLAT2 * DEGRAD
    olon = OLON * DEGRAD
    olat = OLAT * DEGRAD

    sn = math.tan(math.pi * 0.25 + slat2 * 0.5) / math.tan(math.pi * 0.25 + slat1 * 0.5)
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(sn)

    sf = math.tan(math.pi * 0.25 + slat1 * 0.5)
    sf = (sf ** sn) * math.cos(slat1) / sn

    ro = math.tan(math.pi * 0.25 + olat * 0.5)
    ro = re * sf / (ro ** sn)

    ra = math.tan(math.pi * 0.25 + (lat * DEGRAD) * 0.5)
    ra = re * sf / (ra ** sn)

    theta = lon * DEGRAD - olon
    if theta > math.pi:
        theta -= 2.0 * math.pi
    if theta < -math.pi:
        theta += 2.0 * math.pi
    theta *= sn

    x = int(math.floor(ra * math.sin(theta) + XO + 0.5))
    y = int(math.floor(ro - ra * math.cos(theta) + YO + 0.5))
    return x, y


def _kma_base_datetime_kst(now: dt.datetime) -> Tuple[str, str]:
    """
    getVilageFcst의 base_time은 1일 8회(0200,0500,0800,1100,1400,1700,2000,2300). :contentReference[oaicite:4]{index=4}
    그리고 실제 API는 발표 직후가 아니라 약간(예: 10분) 뒤부터 안정적으로 조회되는 경우가 많아서 buffer를 둔다. :contentReference[oaicite:5]{index=5}
    """
    release_times = ["2300", "2000", "1700", "1400", "1100", "0800", "0500", "0200"]
    buffer_minutes = 10

    for t in release_times:
        hh = int(t[:2])
        mm = int(t[2:])
        candidate = now.replace(hour=hh, minute=mm, second=0, microsecond=0) + dt.timedelta(minutes=buffer_minutes)
        if now >= candidate:
            base = candidate - dt.timedelta(minutes=buffer_minutes)
            return base.strftime("%Y%m%d"), base.strftime("%H%M")

    # 02:10(버퍼) 이전이면 전날 23:00 사용
    prev_day = (now - dt.timedelta(days=1)).replace(hour=23, minute=0, second=0, microsecond=0)
    return prev_day.strftime("%Y%m%d"), "2300"


def _safe_get(d: dict, path: List[str], default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

# [추가] requests 재시도 세션(공공 API 타임아웃/일시 장애 대비)
def _requests_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def fetch_vilage_fcst_weather(
    lat: float,
    lon: float,
    timeout: int = 10,
) -> dict:
    """
    프록시(KMA_VILAGE_BASE_URL)를 통해 단기예보(getVilageFcst)를 호출하고,
    '지금 시각에 가장 가까운 1시간 예보'를 뽑아 weather dict로 변환한다.
    """

    kst = ZoneInfo("Asia/Seoul")
    now = dt.datetime.now(tz=kst)
    base_date, base_time = _kma_base_datetime_kst(now)
    nx, ny = latlon_to_grid(lat, lon)

    params = {
        "pageNo": "1",
        "numOfRows": "1000",
        "dataType": "JSON",
        "base_date": base_date,
        "base_time": base_time,
        "nx": str(nx),
        "ny": str(ny),
    }

    sess = _requests_session()
    r = sess.get(KMA_VILAGE_BASE_URL, params=params, timeout=(5, 25))

    r.raise_for_status()
    payload = r.json()

    result_code = _safe_get(payload, ["response", "header", "resultCode"])
    if result_code != "00":
        msg = _safe_get(payload, ["response", "header", "resultMsg"], "UNKNOWN_ERROR")
        raise RuntimeError(f"KMA API error: {result_code} / {msg}")

    items = _safe_get(payload, ["response", "body", "items", "item"], [])
    if not items:
        raise RuntimeError("KMA API returned no items.")

    now_naive = now.replace(tzinfo=None)
    candidates = {}
    for it in items:
        try:
            fcst_dt = dt.datetime.strptime(it["fcstDate"] + it["fcstTime"], "%Y%m%d%H%M")
        except Exception:
            continue
        if fcst_dt < (now_naive - dt.timedelta(hours=2)):
            continue
        key = fcst_dt.strftime("%Y%m%d%H%M")
        candidates.setdefault(key, []).append(it)

    if not candidates:
        raise RuntimeError("예보 후보 슬롯을 찾지 못했다.")

    sorted_slots = sorted(
        candidates.keys(),
        key=lambda k: abs((dt.datetime.strptime(k, "%Y%m%d%H%M") - now_naive).total_seconds())
        if dt.datetime.strptime(k, "%Y%m%d%H%M") >= now_naive
        else 10**18
    )
    chosen_key = sorted_slots[0]
    chosen_dt = dt.datetime.strptime(chosen_key, "%Y%m%d%H%M")
    slot_items = candidates[chosen_key]

    vals = {}
    for it in slot_items:
        cat = it.get("category")
        v = it.get("fcstValue")
        if cat and v is not None:
            vals[cat] = v

    temp_c = None
    if "TMP" in vals:
        try:
            temp_c = int(float(vals["TMP"]))
        except Exception:
            temp_c = None

    pty = str(vals.get("PTY", "0")).strip()
    precip = "없음"
    if pty == "0":
        precip = "없음"
    elif pty == "1":
        precip = "비"
    elif pty == "2":
        precip = "비/눈"
    elif pty == "3":
        precip = "눈"
    elif pty == "4":
        precip = "비"

    wind_level = 3
    if "WSD" in vals:
        try:
            wsd = float(vals["WSD"])
            wind_level = int(max(0, min(10, round(wsd * 1.2))))
        except Exception:
            wind_level = 3

    pop = None
    if "POP" in vals:
        try:
            pop = int(float(vals["POP"]))
        except Exception:
            pop = None

    weather = {
        "temp_c": temp_c if temp_c is not None else 10,
        "precip": precip,
        "wind_level": wind_level,
        "source": "KMA:getVilageFcst(PROXY)",
        "base_date": base_date,
        "base_time": base_time,
        "fcst_at": chosen_dt.strftime("%Y-%m-%d %H:%M"),
        "nx": nx,
        "ny": ny,
        "pop_percent": pop,
    }
    return weather





# =========================
# DB LAYER
# =========================
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = db()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS user_profile (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            age INTEGER,
            gender TEXT,
            closet_style TEXT,
            location_allowed INTEGER,
            created_at TEXT
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS closet_items (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT,
            color TEXT,
            length TEXT,
            fit TEXT,
            flashiness INTEGER, -- 0~10
            tags_json TEXT,     -- ["봄","미니멀","캐주얼"] etc
            owned INTEGER DEFAULT 1,  -- ✅ [추가] 1=있음, 0=없음
            created_at TEXT
        )
        """
    )

    # ✅ [추가] owned 컬럼이 없으면 추가 (기존 DB 마이그레이션)
cols = cur.execute("PRAGMA table_info(closet_items)").fetchall()
col_names = {c[1] for c in cols}
if "owned" not in col_names:
    cur.execute("ALTER TABLE closet_items ADD COLUMN owned INTEGER DEFAULT 1")

    try:
        cur.execute("ALTER TABLE closet_items ADD COLUMN owned INTEGER DEFAULT 1")
    except Exception:
        pass

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS outfits (
            id TEXT PRIMARY KEY,
            date TEXT NOT NULL,          -- YYYY-MM-DD
            situation TEXT NOT NULL,
            title TEXT NOT NULL,
            items_json TEXT NOT NULL,    -- [{"id":"...","name":"..."}]
            notes TEXT,                  -- model rationale or short memo
            weather_json TEXT,           -- placeholder
            created_at TEXT
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS outfit_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            outfit_id TEXT NOT NULL,
            verdict TEXT NOT NULL,       -- good / bad
            bad_reason TEXT,             -- too_hot / too_cold / not_suitable / not_pretty
            created_at TEXT,
            FOREIGN KEY(outfit_id) REFERENCES outfits(id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS preference_memory (
            key TEXT PRIMARY KEY,        -- e.g., "fit:오버핏", "color:블랙"
            score REAL NOT NULL
        )
        """
    )
    # ✅ [추가] 기존 DB에 owned 컬럼이 없으면 추가(마이그레이션)
    try:
        cur.execute("ALTER TABLE closet_items ADD COLUMN owned INTEGER DEFAULT 1")
    except Exception:
        pass

    conn.commit()
    conn.close()


def get_profile() -> Optional[dict]:
    conn = db()
    cur = conn.cursor()
    row = cur.execute("SELECT * FROM user_profile WHERE id=1").fetchone()
    conn.close()
    return dict(row) if row else None


def upsert_profile(age: int, gender: str, closet_style: str, location_allowed: bool) -> None:
    conn = db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO user_profile(id, age, gender, closet_style, location_allowed, created_at)
        VALUES (1, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            age=excluded.age,
            gender=excluded.gender,
            closet_style=excluded.closet_style,
            location_allowed=excluded.location_allowed
        """
        ,
        (age, gender, closet_style, 1 if location_allowed else 0, dt.datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def list_closet_items() -> List[dict]:
    conn = db()
    cur = conn.cursor()
    rows = cur.execute("SELECT * FROM closet_items ORDER BY created_at DESC").fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        d["tags"] = json.loads(d["tags_json"]) if d.get("tags_json") else []
        out.append(d)
    return out


def insert_closet_items(items: List[dict]) -> None:
    conn = db()
    cur = conn.cursor()
    now = dt.datetime.now().isoformat()
    for it in items:
        cur.execute(
            """
            INSERT OR REPLACE INTO closet_items
            (id, name, category, color, length, fit, flashiness, tags_json, owned, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                it["id"],
                it["name"],
                it.get("category"),
                it.get("color"),
                it.get("length"),
                it.get("fit"),
                int(it.get("flashiness", 3)),
                json.dumps(it.get("tags", []), ensure_ascii=False),
                int(it.get("owned", 1)),
                now,
            ),
        )

    conn.commit()
    conn.close()


def delete_item(item_id: str) -> None:
    conn = db()
    cur = conn.cursor()
    cur.execute("DELETE FROM closet_items WHERE id=?", (item_id,))
    conn.commit()
    conn.close()


def update_item(item_id: str, patch: dict) -> None:
    allowed = ["name", "category", "color", "length", "fit", "flashiness", "tags_json", "owned"]
    sets = []
    vals = []
    for k, v in patch.items():
        if k not in allowed:
            continue
        sets.append(f"{k}=?")
        vals.append(v)
    if not sets:
        return
    vals.append(item_id)
    conn = db()
    cur = conn.cursor()
    cur.execute(f"UPDATE closet_items SET {', '.join(sets)} WHERE id=?", tuple(vals))
    conn.commit()
    conn.close()

def set_item_owned(item_id: str, owned: bool) -> None:
    conn = db()
    cur = conn.cursor()
    cur.execute("UPDATE closet_items SET owned=? WHERE id=?", (1 if owned else 0, item_id))
    conn.commit()
    conn.close()


def list_owned_closet_items() -> List[dict]:
    conn = db()
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT * FROM closet_items WHERE IFNULL(owned,1)=1 ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        d["tags"] = json.loads(d["tags_json"]) if d.get("tags_json") else []
        out.append(d)
    return out


def save_outfit(outfit: dict) -> None:
    conn = db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO outfits
        (id, date, situation, title, items_json, notes, weather_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            outfit["id"],
            outfit["date"],
            outfit["situation"],
            outfit["title"],
            json.dumps(outfit["items"], ensure_ascii=False),
            outfit.get("notes"),
            json.dumps(outfit.get("weather", {}), ensure_ascii=False),
            dt.datetime.now().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def list_outfits(limit: int = 50) -> List[dict]:
    conn = db()
    cur = conn.cursor()
    rows = cur.execute("SELECT * FROM outfits ORDER BY date DESC, created_at DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        d["items"] = json.loads(d["items_json"])
        d["weather"] = json.loads(d["weather_json"]) if d.get("weather_json") else {}
        out.append(d)
    return out


def add_feedback(outfit_id: str, verdict: str, bad_reason: Optional[str] = None) -> None:
    conn = db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO outfit_feedback(outfit_id, verdict, bad_reason, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (outfit_id, verdict, bad_reason, dt.datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def list_feedback_for_outfit(outfit_id: str) -> List[dict]:
    conn = db()
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT verdict, bad_reason, created_at FROM outfit_feedback WHERE outfit_id=? ORDER BY id DESC",
        (outfit_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def bump_preference(keys: List[str], delta: float) -> None:
    if not keys:
        return
    conn = db()
    cur = conn.cursor()
    for k in keys:
        row = cur.execute("SELECT score FROM preference_memory WHERE key=?", (k,)).fetchone()
        if row:
            cur.execute("UPDATE preference_memory SET score=? WHERE key=?", (float(row["score"]) + delta, k))
        else:
            cur.execute("INSERT INTO preference_memory(key, score) VALUES (?, ?)", (k, delta))
    conn.commit()
    conn.close()


def get_preference_summary(top_n: int = 12) -> List[Tuple[str, float]]:
    conn = db()
    cur = conn.cursor()
    rows = cur.execute("SELECT key, score FROM preference_memory ORDER BY score DESC LIMIT ?", (top_n,)).fetchall()
    conn.close()
    return [(r["key"], float(r["score"])) for r in rows]


# =========================
# GPT HELPERS
# =========================
def openai_client(api_key: str):
    if OpenAI is None:
        raise RuntimeError("openai 패키지가 설치되지 않았다. `pip install openai`가 필요하다.")
    return OpenAI(api_key=api_key)


def safe_json_from_model(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start_obj = text.find("{")
    start_arr = text.find("[")
    candidates = [p for p in [start_obj, start_arr] if p != -1]
    if not candidates:
        raise ValueError("JSON을 찾지 못했다.")
    start = min(candidates)
    end = max(text.rfind("}"), text.rfind("]"))
    if end == -1 or end <= start:
        raise ValueError("JSON 경계가 이상하다.")
    return json.loads(text[start : end + 1])


def gpt_generate_initial_closet(
    api_key: str,
    age: int,
    gender: str,
    closet_style: str,
    n_items: int = 30,
    model: str = "gpt-5-mini",
) -> List[dict]:
    client = openai_client(api_key)
    prompt = f"""
너는 '오늘 뭐 입지' 앱의 옷장 초기 데이터를 생성하는 도우미다.
사용자 프로필:
- 나이: {age}
- 성별: {gender}
- 옷장 스타일: {closet_style}

요구사항:
- 현실적인 데일리 옷장 아이템 {n_items}개를 생성한다.
- 각 아이템은 반드시 아래 JSON 스키마를 따른다.
- id는 "itm_001"처럼 3자리 넘버링으로 유일해야 한다.
- category는 다음 중 하나: "top","bottom","outer","shoes","bag","accessory"
- color는 영어 소문자(black, white, navy, beige, gray, brown, blue, green, red 등)로.
- length는 "short","regular","long" 중 하나.
- fit은 "slim","regular","oversized","wide" 중 하나.
- flashiness는 0~10 정수.
- tags는 한국어 키워드 2~4개(계절/무드/스타일 등).

출력은 JSON만. 다른 문장 금지.
"""
    resp = client.responses.create(
        model=model,
        input=prompt,
    )
    data = safe_json_from_model(resp.output_text)
    if not isinstance(data, list) or len(data) < 10:
        raise ValueError("옷장 생성 결과 형식이 올바르지 않다.")
    return data[:n_items]


def gpt_recommend_outfit(
    api_key: str,
    profile: dict,
    closet_items: List[dict],
    situation: str,
    weather: dict,
    preference_summary: List[Tuple[str, float]],
    model: str = "gpt-5-mini",
) -> dict:
    client = openai_client(api_key)

    pref_text = "\n".join([f"- {k}: {s:.2f}" for k, s in preference_summary]) or "- (아직 학습된 선호가 거의 없음)"

    closet_compact = [
        {
            "id": it["id"],
            "name": it["name"],
            "category": it.get("category"),
            "color": it.get("color"),
            "length": it.get("length"),
            "fit": it.get("fit"),
            "flashiness": it.get("flashiness"),
            "tags": it.get("tags", []),
        }
        for it in closet_items
    ]

    prompt = f"""
너는 '오늘 뭐 입지' 앱의 코디 추천 엔진이다.
반드시 사용자의 '내 옷장' 아이템에서만 골라 코디를 구성한다(없는 옷 생성 금지).

[사용자 프로필]
- 나이: {profile.get("age")}
- 성별: {profile.get("gender")}
- 옷장 스타일: {profile.get("closet_style")}

[오늘의 상황]
- {situation}

[오늘의 날씨]
- {json.dumps(weather, ensure_ascii=False)}

[사용자가 좋아하는 경향(점수 높을수록 선호)]
{pref_text}

[내 옷장 아이템 목록(JSON)]
{json.dumps(closet_compact, ensure_ascii=False)}

요구사항:
- 상황/날씨/선호를 반영한 "오늘의 코디" 1세트를 만든다.
- 상의/하의/아우터(필요시)/신발(가능하면)/가방 또는 악세서리(가능하면)로 구성한다.
- 꼭 필요한 경우가 아니라면 과하게 화려하지 않게.
- 출력은 아래 JSON 스키마만.

출력 JSON 스키마:
{{
  "title": "코디 이름",
  "items": [{{"id":"...", "name":"..."}}],
  "notes": "추천 이유 및 착용 팁 2~3문장"
}}

다른 문장 금지. JSON만 출력.
"""
    resp = client.responses.create(model=model, input=prompt)
    data = safe_json_from_model(resp.output_text)

    if not isinstance(data, dict) or "items" not in data:
        raise ValueError("코디 추천 결과 형식이 올바르지 않다.")

    closet_ids = {it["id"] for it in closet_items}
    filtered = []
    for it in data.get("items", []):
        if isinstance(it, dict) and it.get("id") in closet_ids:
            filtered.append({"id": it["id"], "name": it.get("name", "")})
    if len(filtered) < 2:
        raise ValueError("옷장 기반 아이템 매칭에 실패했다.")
    data["items"] = filtered
    return data


# =========================
# PREFERENCE UPDATE LOGIC
# =========================
def preference_keys_from_outfit_items(items: List[dict], closet_lookup: Dict[str, dict]) -> List[str]:
    keys: List[str] = []
    for itref in items:
        iid = itref["id"]
        it = closet_lookup.get(iid)
        if not it:
            continue
        if it.get("fit"):
            keys.append(f"fit:{it['fit']}")
        if it.get("color"):
            keys.append(f"color:{it['color']}")
        if it.get("category"):
            keys.append(f"cat:{it['category']}")
        for t in it.get("tags", [])[:2]:
            keys.append(f"tag:{t}")
    seen = set()
    uniq = []
    for k in keys:
        if k not in seen:
            uniq.append(k)
            seen.add(k)
    return uniq


# =========================
# UI COMPONENTS
# =========================
def render_item_card(it: dict) -> None:
    tags = it.get("tags", [])
    st.markdown(
        f"""
        <div style="border:1px solid #e6e6e6; border-radius:14px; padding:14px; background:#ffffff;">
          <div style="font-size:16px; font-weight:700; margin-bottom:6px;">{it['name']}</div>
          <div style="font-size:13px; color:#555;">
            카테고리: {it.get('category','-')} · 컬러: {it.get('color','-')} · 기장: {it.get('length','-')} · 핏: {it.get('fit','-')} · 화려함: {it.get('flashiness','-')}/10
          </div>
          <div style="margin-top:8px; font-size:12px; color:#777;">
            태그: {" · ".join(tags) if tags else "-"}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_outfit_card(outfit: dict) -> None:
    items = outfit["items"]
    st.markdown(
        f"""
        <div style="border:1px solid #e6e6e6; border-radius:16px; padding:16px; background:#ffffff;">
          <div style="font-size:18px; font-weight:800; margin-bottom:8px;">{outfit["title"]}</div>
          <div style="font-size:13px; color:#666; margin-bottom:10px;">상황: {outfit["situation"]} · 날짜: {outfit["date"]}</div>
          <div style="font-size:14px; margin-bottom:8px;">
            {"<br/>".join([f"• {x['name']}" for x in items])}
          </div>
          <div style="font-size:13px; color:#444; margin-top:10px;">{outfit.get("notes","")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_outfit_card_with_toggles_buffered(
    outfit: dict,
    closet_lookup: Dict[str, dict],
) -> List[str]:
    """
    코디 카드 안에서 아이템별 '있어/없어' 토글을 보여준다.
    - 토글은 즉시 DB 반영 X (세션에만 저장)
    - 반환값: 사용자가 '없어'로 체크한 item_id 리스트
    """

    st.markdown(
        """
        <style>
        .outfit-card {
            border: 1px solid #e6e6e6;
            border-radius: 16px;
            padding: 16px;
            background: #ffffff;
            margin-bottom: 12px;
        }
        .outfit-title {
            font-size: 18px;
            font-weight: 800;
            margin-bottom: 8px;
        }
        .outfit-meta {
            font-size: 13px;
            color: #666;
            margin-bottom: 12px;
        }
        .outfit-item {
            font-size: 15.5px;
            line-height: 1.65;
            margin: 6px 0;
        }
        .outfit-item-name { font-weight: 650; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    outfit_id = outfit["id"]
    state_key = f"owned_buffer_{outfit_id}"
    if state_key not in st.session_state:
        st.session_state[state_key] = {}  # {item_id: True/False}

    st.markdown('<div class="outfit-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="outfit-title">{outfit["title"]}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="outfit-meta">상황: {outfit["situation"]} · 날짜: {outfit["date"]}</div>',
        unsafe_allow_html=True,
    )
    # ✅ [수정] 토글 열 헤더(토글과 같은 컬럼 비율로 맞춤)
    rowL, rowR = st.columns([0.80, 0.20], gap="small", vertical_alignment="center")
    with rowL:
        st.write("")  # 왼쪽은 비움
    with rowR:
        st.markdown(
            '<div style="text-align:left; font-size:12px; color:#666; font-weight:700; padding-left:6px;">이 옷 있어!</div>',
            unsafe_allow_html=True,
        )

    # ✅ 헤더 아래에서 실제 토글 목록 렌더링 (헤더 with rowR 밖!)
    missing_ids: List[str] = []

    for itref in outfit.get("items", []):
        iid = itref["id"]
        name = itref.get("name", "")

        owned_db = int(closet_lookup.get(iid, {}).get("owned", 1)) == 1
        owned_now = st.session_state[state_key].get(iid, owned_db)

        left, right = st.columns([0.80, 0.20], gap="small", vertical_alignment="center")

        with left:
            st.markdown(
                f'<div class="outfit-item">• <span class="outfit-item-name">{name}</span></div>',
                unsafe_allow_html=True,
            )

        with right:
            toggled = st.toggle(
                "있어/없어",
                value=owned_now,
                key=f"owned_toggle_{outfit_id}_{iid}",
                label_visibility="collapsed",
            )
            st.session_state[state_key][iid] = toggled

        if st.session_state[state_key][iid] is False:
            missing_ids.append(iid)

    st.markdown("</div>", unsafe_allow_html=True)

    return missing_ids


def onboarding_screen() -> None:
    st.title(APP_TITLE)
    st.caption("귀찮은 사람들을 위한 코디 추천 어플이다. 최초 1회만 물어본다.")

    # (중요) 온보딩 화면에서도 위치 컴포넌트를 렌더링해줘야 브라우저 권한 팝업이 뜬다.
    # 체크박스를 켜면 즉시 geolocation 컴포넌트가 렌더링되면서 허용 요청이 뜬다.

    with st.form("onboarding_form", clear_on_submit=False):
        age = st.number_input("나이", min_value=10, max_value=80, value=22, step=1)
        gender = st.selectbox("성별", ["여성", "남성", "논바이너리/기타", "비공개"])
        closet_style = st.selectbox(
            "옷장의 스타일",
            ["편한 게 최고", "무조건 깔끔단정", "스트릿/힙", "미니멀", "화려하게", "빈티지", "스포티", "기타"],
        )
        location_allowed = st.checkbox("위치 정보 허용(허용하면 자동으로 날씨를 불러온다)", value=True)

        # ✅ 여기: 체크하면 바로 브라우저 권한 요청이 뜨도록 컴포넌트를 렌더링
        loc = None
        if location_allowed:
            st.markdown("✅ 브라우저에서 위치 권한을 **허용**해라. (버튼 누를 필요 없다)")
            loc = streamlit_geolocation()

            if loc and loc.get("latitude") and loc.get("longitude"):
                st.success(f"위치 확인됨: lat={loc['latitude']}, lon={loc['longitude']}")
            else:
                st.info("아직 위치값을 못 받았다. 브라우저 팝업에서 허용했는지 확인해라.")

        submitted = st.form_submit_button("시작하기")

    if submitted:
        upsert_profile(int(age), gender, closet_style, bool(location_allowed))

        # ✅ 여기: 온보딩 끝날 때 위경도를 세션에 저장해둔다.
        if location_allowed and loc and loc.get("latitude") and loc.get("longitude"):
            st.session_state["geo_lat"] = str(loc["latitude"])
            st.session_state["geo_lon"] = str(loc["longitude"])

        st.success("온보딩 완료다. 이제 자동으로 날씨를 불러온다.")
        st.session_state["onboarded"] = True
        st.rerun()


def ensure_initial_closet(profile: dict, api_key: str) -> None:
    items = list_closet_items()
    if len(items) >= 10:
        return

    st.info("옷장을 초기 세팅하는 중이다. (처음 한 번만)")
    if not api_key:
        sample = make_fallback_closet(profile)
        insert_closet_items(sample)
        st.warning("API 키가 없어서 샘플 옷장으로 채웠다. 사이드바에 키 넣으면 더 정확해진다.")
        return

    try:
        with st.spinner("GPT로 옷장 30개 생성 중..."):
            generated = gpt_generate_initial_closet(
                api_key=api_key,
                age=int(profile["age"]),
                gender=str(profile["gender"]),
                closet_style=str(profile["closet_style"]),
                n_items=30,
                model="gpt-5-mini",
            )
        insert_closet_items(generated)
        st.success("옷장 초기 아이템 30개 등록 완료다.")
    except Exception as e:
        st.error(f"옷장 생성에 실패했다: {e}")
        sample = make_fallback_closet(profile)
        insert_closet_items(sample)
        st.warning("일단 샘플 옷장으로 채웠다. 나중에 다시 시도하면 된다.")


def make_fallback_closet(profile: dict) -> List[dict]:
    base = [
        ("기본핏 흰 티셔츠", "top", "white", "regular", "regular", 1, ["베이직", "사계절"]),
        ("회색 맨투맨", "top", "gray", "regular", "oversized", 2, ["캐주얼", "봄가을"]),
        ("검정 슬랙스", "bottom", "black", "long", "regular", 1, ["깔끔", "데일리"]),
        ("연청 데님팬츠", "bottom", "blue", "long", "regular", 2, ["캐주얼", "데님"]),
        ("블랙 후드집업", "outer", "black", "regular", "regular", 2, ["레이어드", "봄가을"]),
        ("베이지 트렌치코트", "outer", "beige", "long", "regular", 3, ["깔끔", "간절기"]),
        ("흰 스니커즈", "shoes", "white", "regular", "regular", 1, ["만능", "데일리"]),
        ("검정 로퍼", "shoes", "black", "regular", "regular", 2, ["단정", "오피스"]),
        ("크로스백", "bag", "black", "regular", "regular", 2, ["실용", "데일리"]),
        ("심플 시계", "accessory", "silver", "regular", "regular", 2, ["미니멀", "포인트"]),
    ]
    items = []
    for i in range(1, 31):
        name, cat, color, length, fit, flash, tags = base[(i - 1) % len(base)]
        items.append(
            {
                "id": f"itm_{i:03d}",
                "name": name if i <= len(base) else f"{name} (변형 {i})",
                "category": cat,
                "color": color,
                "length": length,
                "fit": fit,
                "flashiness": flash,
                "tags": tags,
            }
        )
    return items


# =========================
# [수정] sidebar_controls: 기상청 키+위경도 입력 & 호출 버튼 추가
# (기존 '임시 입력'은 그대로 두고, KMA가 성공하면 그 값을 override)
# =========================
def _maybe_autofetch_weather(
    lat: str,
    lon: str,
    temp: int,
    rain: str,
    wind: int,
) -> None:
    if not lat.strip() or not lon.strip():
        return
    try:
        _lat = float(lat.strip())
        _lon = float(lon.strip())
    except ValueError:
        return


    last_fetch = st.session_state.get("kma_last_fetch")
    now = dt.datetime.now()
        # [추가] 방금 실패했다면(3분) 자동 재시도하지 않음
    last_fail = st.session_state.get("kma_last_fail")
    if last_fail:
        try:
            fail_at = dt.datetime.fromisoformat(last_fail["at"])
        except Exception:
            fail_at = None
        if fail_at and (now - fail_at).total_seconds() < 3 * 60:
            return

    if last_fetch:
        try:
            last_at = dt.datetime.fromisoformat(last_fetch["at"])
        except Exception:
            last_at = None
        if (
            last_at
            and abs((now - last_at).total_seconds()) < 30 * 60
            and float(last_fetch.get("lat", 0)) == _lat
            and float(last_fetch.get("lon", 0)) == _lon
        ):
            return

    try:
        with st.spinner("기상청 단기예보 자동 불러오는 중..."):
            w = fetch_vilage_fcst_weather(lat=_lat, lon=_lon)

        st.session_state["weather_live"] = {
            "temp_c": int(w.get("temp_c", temp)),
            "precip": str(w.get("precip", rain)),
            "wind_level": int(w.get("wind_level", wind)),
            "source": w.get("source"),
            "base_date": w.get("base_date"),
            "base_time": w.get("base_time"),
            "fcst_at": w.get("fcst_at"),
            "pop_percent": w.get("pop_percent"),
            "nx": w.get("nx"),
            "ny": w.get("ny"),
        }
        st.session_state["kma_last_fetch"] = {"lat": _lat, "lon": _lon, "at": now.isoformat()}
        st.sidebar.success(f"자동 반영: {st.session_state['weather_live'].get('fcst_at')} 예보 기준")
    except Exception as e:
        st.sidebar.error(f"자동 기상청 호출 실패: {e}")
        st.sidebar.info("임시 입력 값으로 계속 진행한다.")
        st.sidebar.info("임시 입력 값으로 계속 진행한다.")

def sidebar_controls(profile: dict) -> Dict[str, Any]:
    st.sidebar.header("설정")
    api_key = st.sidebar.text_input(
        "OpenAI API 키",
        type="password",
        help="키는 로컬/서버에 저장되지 않게 설계하는 편이 안전하다.",
    )
    st.sidebar.divider()

    # ---- 위치/날씨: 버튼 없이 자동 ----
    st.sidebar.subheader("내 위치 & 자동 날씨")

    # 온보딩에서 위치 허용한 사람만 자동 위치를 요청/갱신한다.
    if int(profile.get("location_allowed", 0)) == 1:
        loc = streamlit_geolocation()
        if loc and loc.get("latitude") and loc.get("longitude"):
            st.session_state["geo_lat"] = str(loc["latitude"])
            st.session_state["geo_lon"] = str(loc["longitude"])
            st.sidebar.success("위치 자동 감지됨")
        else:
            st.sidebar.info("위치값을 아직 못 받았다. 브라우저에서 위치 허용했는지 확인해라.")

    # 세션에 저장된 위경도가 있으면 자동으로 날씨 호출
    lat = st.session_state.get("geo_lat", "")
    lon = st.session_state.get("geo_lon", "")

    # ---- 임시 입력(폴백) ----
    st.sidebar.divider()
    st.sidebar.subheader("날씨(임시 입력)")
    st.sidebar.caption("자동 날씨를 못 가져오면 이 값을 쓴다.")
    temp = st.sidebar.slider("오늘 체감온도(°C)", min_value=-10, max_value=35, value=10, step=1)
    rain = st.sidebar.selectbox("강수", ["없음", "비", "눈", "비/눈"], index=0)
    wind = st.sidebar.slider("바람(체감 영향)", min_value=0, max_value=10, value=3, step=1)

    # weather_live 초기값
    if "weather_live" not in st.session_state:
        st.session_state["weather_live"] = {"temp_c": temp, "precip": rain, "wind_level": wind}

    # ✅ 핵심: 버튼 없이 자동 fetch
    _maybe_autofetch_weather(lat, lon, temp, rain, wind)

    return {"api_key": api_key, "weather": st.session_state["weather_live"]}



def tab_analysis(weather: dict) -> None:
    st.subheader("분석")
    st.caption("현재 반영된 날씨와 학습된 취향을 요약해서 보여준다.")

    st.markdown("#### 현재 반영된 날씨")
    st.json(weather)

    st.markdown("#### 학습된 취향(요약)")
    prefs = get_preference_summary()
    if prefs:
        for k, s in prefs:
            st.write(f"- {k} : {s:.2f}")
    else:
        st.write("아직 데이터가 없다.")



def tab_closet() -> None:
    st.subheader("내 옷장")
    def tab_closet() -> None:
    st.subheader("내 옷장")

    # ✅ [추가] 새 아이템 추가
    with st.expander("➕ 새 아이템 추가", expanded=False):
        with st.form("add_item_form", clear_on_submit=True):
            name = st.text_input("아이템명", placeholder="예: 검정 니트")
            category = st.selectbox("카테고리", ["top", "bottom", "outer", "shoes", "bag", "accessory"])
            color = st.text_input("컬러(영문 소문자)", placeholder="black / white / navy ...")
            length = st.selectbox("기장", ["short", "regular", "long"], index=1)
            fit = st.selectbox("핏", ["slim", "regular", "oversized", "wide"], index=1)
            flash = st.slider("화려함(0~10)", 0, 10, 2)
            tags_str = st.text_input("태그(쉼표로 구분)", placeholder="예: 데일리,미니멀,봄가을")

            submitted = st.form_submit_button("추가하기")

        if submitted:
            if not name.strip():
                st.error("아이템명은 필수다.")
                st.stop()

            # ✅ id 자동 생성
            new_id = f"itm_user_{dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

            new_item = {
                "id": new_id,
                "name": name.strip(),
                "category": category,
                "color": color.strip().lower(),
                "length": length,
                "fit": fit,
                "flashiness": int(flash),
                "tags": [t.strip() for t in tags_str.split(",") if t.strip()],
                "owned": 1,  # ✅ 기본: 있음
            }

            insert_closet_items([new_item])
            st.success("추가했다.")
            st.rerun()

    items = list_closet_items()
    if not items:
        st.info("옷장이 비어 있다.")
        return

    cols = st.columns(3)
    ...

    items = list_closet_items()
    if not items:
        st.info("옷장이 비어 있다.")
        return

    cols = st.columns(3)
    for idx, it in enumerate(items):
        with cols[idx % 3]:
            render_item_card(it)
            with st.expander("수정/삭제", expanded=False):
                new_name = st.text_input("아이템명", value=it["name"], key=f"name_{it['id']}")
                category = st.selectbox(
                    "카테고리",
                    ["top", "bottom", "outer", "shoes", "bag", "accessory"],
                    index=["top","bottom","outer","shoes","bag","accessory"].index(it.get("category") or "top"),
                    key=f"cat_{it['id']}",
                )
                color = st.text_input("컬러(영문 소문자)", value=it.get("color") or "", key=f"color_{it['id']}")
                length = st.selectbox(
                    "기장",
                    ["short", "regular", "long"],
                    index=["short","regular","long"].index(it.get("length") or "regular"),
                    key=f"len_{it['id']}",
                )
                fit = st.selectbox(
                    "핏",
                    ["slim", "regular", "oversized", "wide"],
                    index=["slim","regular","oversized","wide"].index(it.get("fit") or "regular"),
                    key=f"fit_{it['id']}",
                )
                flash = st.slider("화려함(0~10)", 0, 10, int(it.get("flashiness") or 2), key=f"flash_{it['id']}")
                tags_str = st.text_input(
                    "태그(쉼표로 구분)",
                    value=",".join(it.get("tags", [])),
                    key=f"tags_{it['id']}",
                )

                c1, c2 = st.columns(2)
                with c1:
                    if st.button("저장", key=f"save_{it['id']}"):
                        patch = {
                            "name": new_name.strip(),
                            "category": category,
                            "color": color.strip().lower(),
                            "length": length,
                            "fit": fit,
                            "flashiness": int(flash),
                            "tags_json": json.dumps(
                                [t.strip() for t in tags_str.split(",") if t.strip()],
                                ensure_ascii=False,
                            ),
                        }
                        update_item(it["id"], patch)
                        st.success("저장했다.")
                        st.rerun()
                with c2:
                    if st.button("삭제", key=f"del_{it['id']}"):
                        delete_item(it["id"])
                        st.warning("삭제했다.")
                        st.rerun()


def tab_recommend(profile: dict, api_key: str, weather: dict) -> None:
    st.subheader("코디 추천")
    st.caption("기본 메인 화면이다. 상황을 고르면 옷장 기반으로 코디를 뽑는다.")

    if "current_outfit" not in st.session_state:
        st.session_state["current_outfit"] = None

    situation = st.selectbox(
        "오늘의 상황",
        ["학교", "데이트", "직장", "피크닉", "운동", "모임", "기타"],
        key="situation",
    )

    # ✅ 상황 바뀌면 코디 초기화 후 rerun
    prev = st.session_state.get("prev_situation")
    if prev is None:
        st.session_state["prev_situation"] = situation
    elif prev != situation:
        st.session_state["prev_situation"] = situation
        st.session_state["current_outfit"] = None
        st.rerun()



    if "current_outfit" not in st.session_state:
        st.session_state["current_outfit"] = None

    all_items = list_closet_items()
    closet_lookup = {it["id"]: it for it in all_items}  # UI 토글 기본값용(owned 포함)
    closet_items = [it for it in all_items if int(it.get("owned", 1)) == 1]  # ✅ 추천 후보는 owned=1만
    pref_summary = get_preference_summary()

    def refresh_reco():
        if not closet_items:
            st.error("옷장이 비어 있다. 먼저 옷장을 채워야 한다.")
            return
        if not api_key:
            outfit = simple_rule_based_outfit(situation, closet_items, weather)
            st.session_state["current_outfit"] = outfit
            return
        try:
            with st.spinner("오늘의 코디를 고르는 중이다..."):
                data = gpt_recommend_outfit(
                    api_key=api_key,
                    profile=profile,
                    closet_items=closet_items,
                    situation=situation,
                    weather=weather,
                    preference_summary=pref_summary,
                    model="gpt-5-mini",
                )
            outfit = {
                "id": f"out_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "date": dt.date.today().isoformat(),
                "situation": situation,
                "title": data["title"],
                "items": data["items"],
                "notes": data.get("notes", ""),
                "weather": weather,
            }
            st.session_state["current_outfit"] = outfit
        except Exception as e:
            st.error(f"추천 실패: {e}")
            outfit = simple_rule_based_outfit(situation, closet_items, weather)
            st.session_state["current_outfit"] = outfit

    if st.session_state["current_outfit"] is None:
        refresh_reco()

    outfit = st.session_state["current_outfit"]
    if outfit:
        # ✅ 카드 안에서 토글 표시(버튼 누를 때까지 DB 반영 X)
        missing_ids = render_outfit_card_with_toggles_buffered(
            outfit=outfit,
            closet_lookup=closet_lookup,
        )

        c1, c_mid, c2 = st.columns([1, 1.2, 1])

        with c1:
            if st.button("별로야"):
                keys = preference_keys_from_outfit_items(outfit["items"], closet_lookup)
                bump_preference(keys, delta=-0.2)
                refresh_reco()
                st.rerun()

        with c_mid:
            disabled = len(missing_ids) == 0
            if st.button("나 이거 없어!", disabled=disabled):
                for iid in missing_ids:
                    update_item(iid, {"owned": 0})

                # 코디 새로고침
                st.session_state["current_outfit"] = None
                st.success(f"없음 처리 {len(missing_ids)}개 반영했다. 코디를 다시 고른다.")
                st.rerun()

        with c2:
            if st.button("이걸로 할래"):
                save_outfit(outfit)
                st.success("오늘의 코디 완료!")
                keys = preference_keys_from_outfit_items(outfit["items"], closet_lookup)
                bump_preference(keys, delta=0.6)
                st.session_state["current_outfit"] = None



def simple_rule_based_outfit(situation: str, closet_items: List[dict], weather: dict) -> dict:
    # ✅ [추가] '없는 옷(owned=0)'은 규칙 추천에서도 제외
    closet_items = [x for x in closet_items if int(x.get("owned", 1)) == 1]
    
    temp = weather.get("temp_c", 10)

    need_outer = temp <= 12 or weather.get("wind_level", 0) >= 7

    def pick(cat: str) -> Optional[dict]:
        xs = [x for x in closet_items if x.get("category") == cat]
        if not xs:
            return None
        idx = (hash(situation + cat + str(temp)) % len(xs))
        return xs[idx]

    picks = []
    top = pick("top")
    bottom = pick("bottom")
    outer = pick("outer") if need_outer else None
    shoes = pick("shoes")
    bag = pick("bag") or pick("accessory")

    for x in [top, bottom, outer, shoes, bag]:
        if x:
            picks.append({"id": x["id"], "name": x["name"]})

    title = f"{situation}용 데일리 코디(임시)"
    notes = "API 키가 없거나 오류라서 임시 규칙 기반으로 골랐다."
    return {
        "id": f"out_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "date": dt.date.today().isoformat(),
        "situation": situation,
        "title": title,
        "items": picks,
        "notes": notes,
        "weather": weather,
    }


def tab_today_collection() -> None:
    st.subheader("오늘의 코디 모음")
    outfits = list_outfits(limit=100)
    if not outfits:
        st.info("아직 저장된 코디가 없다.")
        return

    cols = st.columns(2)
    for i, out in enumerate(outfits):
        with cols[i % 2]:
            render_outfit_card(out)
            fb = list_feedback_for_outfit(out["id"])
            last = fb[0] if fb else None
            if last:
                st.caption(f"최근 피드백: {last['verdict']} / {last.get('bad_reason') or '-'}")

            with st.expander("피드백 남기기", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("좋은 코디였어", key=f"good_{out['id']}"):
                        add_feedback(out["id"], "good", None)
                        closet_items = list_closet_items()
                        lookup = {it["id"]: it for it in closet_items}
                        keys = preference_keys_from_outfit_items(out["items"], lookup)
                        bump_preference(keys, delta=1.0)
                        st.success("반영했다.")
                        st.rerun()

                with c2:
                    if st.button("별로였어", key=f"bad_{out['id']}"):
                        st.session_state[f"ask_reason_{out['id']}"] = True
                        st.rerun()

                if st.session_state.get(f"ask_reason_{out['id']}"):
                    reason = st.radio(
                        "왜 별로였나?",
                        ["너무 더웠어", "너무 추웠어", "상황에 맞지 않았어", "예쁘지 않았어"],
                        key=f"reason_{out['id']}",
                    )
                    reason_map = {
                        "너무 더웠어": "too_hot",
                        "너무 추웠어": "too_cold",
                        "상황에 맞지 않았어": "not_suitable",
                        "예쁘지 않았어": "not_pretty",
                    }
                    if st.button("제출", key=f"submit_reason_{out['id']}"):
                        add_feedback(out["id"], "bad", reason_map[reason])

                        closet_items = list_closet_items()
                        lookup = {it["id"]: it for it in closet_items}
                        keys = preference_keys_from_outfit_items(out["items"], lookup)

                        delta = -1.0 if reason_map[reason] == "not_pretty" else -0.6
                        bump_preference(keys, delta=delta)

                        st.session_state[f"ask_reason_{out['id']}"] = False
                        st.success("피드백 반영했다.")
                        st.rerun()


# =========================
# MAIN
# =========================
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="👕", layout="wide")
    init_db()

    profile = get_profile()
    if "onboarded" not in st.session_state:
        st.session_state["onboarded"] = bool(profile)

    if not st.session_state["onboarded"]:
        onboarding_screen()
        return

    profile = get_profile()
    if not profile:
        onboarding_screen()
        return

    settings = sidebar_controls(profile)
    api_key = settings["api_key"]
    weather = settings["weather"]

    ensure_initial_closet(profile, api_key)
    # ✅ [추가] 온보딩 직후/첫 진입에는 기본 탭을 "코디 추천"으로 보이게 한다.
    # 쿼리 파라미터 tab=reco 를 사용해서, tabs를 만들 때 key로 기본 선택을 유도한다.
    q = st.query_params
    if "tab" not in q:
        st.query_params["tab"] = "reco"
        st.rerun()

    tab_order = ["closet", "reco", "history", "analysis"]
    tab_names = ["내 옷장", "코디 추천", "오늘의 코디 모음", "분석"]

    default_idx = 1  # reco
    try:
        default_idx = tab_order.index(st.query_params.get("tab", "reco"))
    except Exception:
        default_idx = 1

    tabs = st.tabs(tab_names)

    # ✅ [추가] 기본 인덱스 탭이 보이도록, 해당 탭 블록을 먼저 실행(렌더)한다.
    # Streamlit은 탭 “선택” API가 없어서, 렌더 순서로 체감 기본 탭을 맞춘다.
    render_order = list(range(len(tabs)))
    render_order.remove(default_idx)
    render_order = [default_idx] + render_order

    tab_handlers = [
        lambda: tab_closet(),
        lambda: tab_recommend(profile, api_key, weather),
        lambda: tab_today_collection(),
        lambda: tab_analysis(weather),
    ]

    for i in render_order:
        with tabs[i]:
            tab_handlers[i]()
            
if __name__ == "__main__":
    main()




