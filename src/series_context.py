import hashlib
import json
import os
import re
from datetime import datetime, timezone

import requests

from ollama_utils import call_ollama_model


ANILIST_GRAPHQL_URL = "https://graphql.anilist.co"
SERIES_CONTEXT_SCHEMA_VERSION = 1
DEFAULT_SERIES_CACHE_SUBDIR = "YakuZen\\series-cache"
TITLE_NOISE_TOKENS = {
    "AMZN", "AMZNWEBRIP", "WEB", "WEBRIP", "WEBDL", "WEB-DL", "BD", "BDRIP", "BLURAY", "BRRIP",
    "DVDRIP", "HDRIP", "REMUX", "NF", "DSNP", "CR", "ATVP", "FUNI", "HULU", "1080P", "720P",
    "480P", "2160P", "4K", "UHD", "HEVC", "X264", "X265", "H264", "H265", "AAC", "DDP", "AC3",
    "TRUEHD", "DTS", "10BIT", "8BIT", "MULTI", "SUBS", "DUB", "RAW", "BK", "JPN", "ENG",
}
ANILIST_QUERY = """
query ($search: String) {
  Media(search: $search, type: ANIME) {
    id
    siteUrl
    title {
      romaji
      english
      native
    }
    synonyms
    description(asHtml: false)
    studios(isMain: true) {
      nodes {
        name
      }
    }
    characters(sort: [ROLE, RELEVANCE, FAVOURITES_DESC], perPage: 12) {
      edges {
        role
        node {
          name {
            full
            native
            alternative
          }
        }
      }
    }
  }
}
""".strip()


def _utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def infer_series_title(target):
    if os.path.exists(target):
        stem = os.path.splitext(os.path.basename(target))[0]
    else:
        stem = str(target).strip()

    tokens = [token for token in re.split(r"[._\-\s]+", stem) if token]
    title_tokens = []
    year_token = None

    for raw_token in tokens:
        token = raw_token.strip()
        upper_token = token.upper()

        if re.match(r"^(S\d{1,2}E\d{1,3}|E\d{1,3}|EP?\d{1,3}|OVA\d*|OAD\d*|NCOP\d*|NCED\d*)$", upper_token):
            break

        if re.match(r"^(19|20)\d{2}$", token):
            year_token = token
            continue

        if upper_token in TITLE_NOISE_TOKENS:
            if title_tokens:
                break
            continue

        if re.match(r"^\d{3,4}P$", upper_token) or re.match(r"^(Q\d|REMASTER)$", upper_token):
            if title_tokens:
                break
            continue

        title_tokens.append(token)

    if not title_tokens:
        fallback = re.sub(r"[._\-]+", " ", stem).strip()
        return fallback or stem

    title = " ".join(title_tokens).strip()
    if year_token:
        return f"{title} ({year_token})"
    return title


def normalize_series_title(series_title):
    if not series_title:
        return ""
    return re.sub(r"\s*\((19|20)\d{2}\)\s*$", "", str(series_title)).strip()


def slugify_series_title(series_title):
    normalized = normalize_series_title(series_title) or str(series_title).strip()
    slug = re.sub(r"[^a-z0-9]+", "-", normalized.lower()).strip("-")
    return slug or "untitled-series"


def get_global_series_cache_dir():
    if os.name == "nt":
        base_dir = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
        return os.path.join(base_dir, DEFAULT_SERIES_CACHE_SUBDIR)
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return os.path.join(xdg_cache_home, "yakuzen", "series-cache")
    return os.path.expanduser("~/.cache/yakuzen/series-cache")


def get_series_context_cache_path(series_title):
    return os.path.join(get_global_series_cache_dir(), f"{slugify_series_title(series_title)}.json")


def _clean_string_list(values):
    cleaned = []
    seen = set()
    for value in values or []:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
    return cleaned


def normalize_series_context(context, fallback_title=None):
    context = dict(context or {})
    canonical_title_en = str(context.get("canonical_title_en") or fallback_title or "").strip()
    canonical_title_ja = str(context.get("canonical_title_ja") or "").strip()
    aliases = _clean_string_list(context.get("aliases") or [])
    characters = []
    for character in context.get("characters") or []:
        if isinstance(character, str):
            character = {"name_en": character}
        if not isinstance(character, dict):
            continue
        characters.append({
            "name_en": str(character.get("name_en") or "").strip(),
            "name_ja": str(character.get("name_ja") or "").strip(),
            "aliases": _clean_string_list(character.get("aliases") or []),
            "role": str(character.get("role") or "").strip(),
        })

    glossary = []
    for item in context.get("glossary") or []:
        if isinstance(item, str):
            item = {"preferred_en": item}
        if not isinstance(item, dict):
            continue
        glossary.append({
            "preferred_en": str(item.get("preferred_en") or "").strip(),
            "source_ja": str(item.get("source_ja") or "").strip(),
            "aliases": _clean_string_list(item.get("aliases") or []),
            "notes": str(item.get("notes") or "").strip(),
        })

    normalized = {
        "schema_version": SERIES_CONTEXT_SCHEMA_VERSION,
        "canonical_title_en": canonical_title_en,
        "canonical_title_ja": canonical_title_ja,
        "aliases": aliases,
        "characters": characters,
        "organizations": _clean_string_list(context.get("organizations") or []),
        "places": _clean_string_list(context.get("places") or []),
        "glossary": glossary,
        "description": str(context.get("description") or "").strip(),
        "source_refs": context.get("source_refs") or [],
        "locked_fields": _clean_string_list(context.get("locked_fields") or []),
        "last_updated": str(context.get("last_updated") or _utc_now_iso()),
    }
    return normalized


def build_minimal_series_context(series_title):
    return normalize_series_context(
        {
            "canonical_title_en": normalize_series_title(series_title) or str(series_title).strip(),
            "aliases": [normalize_series_title(series_title), str(series_title).strip()],
            "source_refs": [{"provider": "filename", "title": str(series_title).strip()}],
        },
        fallback_title=series_title,
    )


def load_series_context_file(path):
    with open(path, "r", encoding="utf-8") as handle:
        return normalize_series_context(json.load(handle))


def save_series_context_file(context, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(normalize_series_context(context), handle, ensure_ascii=False, indent=2)


def load_cached_series_context(series_title):
    cache_path = get_series_context_cache_path(series_title)
    if not os.path.exists(cache_path):
        return None
    return load_series_context_file(cache_path)


def save_series_context_to_cache(context, series_title=None):
    normalized = normalize_series_context(context, fallback_title=series_title)
    effective_title = normalized.get("canonical_title_en") or series_title or "untitled-series"
    cache_path = get_series_context_cache_path(effective_title)
    save_series_context_file(normalized, cache_path)
    return cache_path


def fetch_series_metadata(series_title):
    response = requests.post(
        ANILIST_GRAPHQL_URL,
        json={"query": ANILIST_QUERY, "variables": {"search": normalize_series_title(series_title)}},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    media = ((payload.get("data") or {}).get("Media")) or {}
    title_info = media.get("title") or {}
    characters = []
    for edge in ((media.get("characters") or {}).get("edges") or []):
        name_info = ((edge.get("node") or {}).get("name")) or {}
        characters.append({
            "name_en": str(name_info.get("full") or "").strip(),
            "name_ja": str(name_info.get("native") or "").strip(),
            "aliases": _clean_string_list(name_info.get("alternative") or []),
            "role": str(edge.get("role") or "").strip(),
        })

    organizations = _clean_string_list(node.get("name") for node in ((media.get("studios") or {}).get("nodes") or []))
    aliases = _clean_string_list(
        [title_info.get("romaji"), title_info.get("english"), title_info.get("native")] + (media.get("synonyms") or [])
    )
    base_context = {
        "canonical_title_en": str(title_info.get("english") or title_info.get("romaji") or normalize_series_title(series_title)).strip(),
        "canonical_title_ja": str(title_info.get("native") or "").strip(),
        "aliases": aliases,
        "characters": characters,
        "organizations": organizations,
        "places": [],
        "glossary": [],
        "description": str(media.get("description") or "").strip(),
        "source_refs": [
            {
                "provider": "AniList",
                "title": normalize_series_title(series_title),
                "url": media.get("siteUrl"),
                "id": media.get("id"),
            }
        ],
        "last_updated": _utc_now_iso(),
    }
    return normalize_series_context(base_context, fallback_title=series_title)


def enrich_series_context(base_context, model_name):
    prompt = (
        "You are preparing structured context for an anime subtitle pipeline.\n"
        "Use ONLY the supplied metadata. Do not invent missing canon.\n"
        "Return ONLY JSON with this exact object shape:\n"
        "{"
        "\"canonical_title_en\":\"\","
        "\"canonical_title_ja\":\"\","
        "\"aliases\":[],"
        "\"characters\":[{\"name_en\":\"\",\"name_ja\":\"\",\"aliases\":[],\"role\":\"\"}],"
        "\"organizations\":[],"
        "\"places\":[],"
        "\"glossary\":[{\"preferred_en\":\"\",\"source_ja\":\"\",\"aliases\":[],\"notes\":\"\"}],"
        "\"description\":\"\""
        "}\n\n"
        f"INPUT METADATA:\n{json.dumps(base_context, ensure_ascii=False, indent=2)}"
    )

    raw_response = call_ollama_model(
        model_name,
        prompt,
        system_prompt="Return only valid JSON. Keep only high-confidence structured facts from the metadata provided.",
        options={"temperature": 0.0},
    )
    match = re.search(r"\{.*\}", raw_response or "", re.DOTALL)
    if not match:
        return base_context
    try:
        enriched = json.loads(match.group(0))
    except json.JSONDecodeError:
        return base_context

    merged = dict(base_context)
    merged.update(enriched)
    merged["source_refs"] = base_context.get("source_refs", [])
    merged["last_updated"] = _utc_now_iso()
    return normalize_series_context(merged, fallback_title=base_context.get("canonical_title_en"))


def resolve_series_context(target, override_path=None, allow_fetch=False, enrich_model=None, force_refresh=False):
    inferred_title = infer_series_title(target)

    if override_path:
        context = load_series_context_file(override_path)
        return normalize_series_context(context, fallback_title=inferred_title), "override", override_path

    if not force_refresh:
        cached_context = load_cached_series_context(inferred_title)
        if cached_context:
            return normalize_series_context(cached_context, fallback_title=inferred_title), "global_cache", get_series_context_cache_path(inferred_title)

    if allow_fetch:
        fetched_context = fetch_series_metadata(inferred_title)
        if enrich_model:
            try:
                fetched_context = enrich_series_context(fetched_context, enrich_model)
            except Exception:
                fetched_context = normalize_series_context(fetched_context, fallback_title=inferred_title)
        cache_path = save_series_context_to_cache(fetched_context, series_title=inferred_title)
        return fetched_context, "fetched", cache_path

    return build_minimal_series_context(inferred_title), "inferred_title", None


def canonical_series_title(context, fallback_title=""):
    context = normalize_series_context(context, fallback_title=fallback_title)
    return (
        context.get("canonical_title_en")
        or context.get("canonical_title_ja")
        or normalize_series_title(fallback_title)
        or str(fallback_title).strip()
    )


def series_context_hash(context, fallback_title=""):
    normalized = normalize_series_context(context, fallback_title=fallback_title)
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def build_series_context_prompt_block(context, fallback_title=""):
    normalized = normalize_series_context(context, fallback_title=fallback_title)
    title_en = normalized.get("canonical_title_en")
    title_ja = normalized.get("canonical_title_ja")
    alias_items = normalized.get("aliases", [])[:6]
    character_items = []
    for character in normalized.get("characters", [])[:8]:
        parts = [part for part in [character.get("name_en"), character.get("name_ja")] if part]
        if parts:
            character_items.append(" / ".join(parts))
    glossary_items = []
    for glossary in normalized.get("glossary", [])[:10]:
        preferred = glossary.get("preferred_en")
        source_ja = glossary.get("source_ja")
        if preferred and source_ja:
            glossary_items.append(f"{source_ja} -> {preferred}")
        elif preferred:
            glossary_items.append(preferred)

    lines = ["ANIME SERIES CONTEXT:"]
    if title_en:
        lines.append(f"- Canonical English title: {title_en}")
    if title_ja:
        lines.append(f"- Canonical Japanese title: {title_ja}")
    if alias_items:
        lines.append(f"- Known aliases: {', '.join(alias_items)}")
    if character_items:
        lines.append(f"- Character names: {', '.join(character_items)}")
    if glossary_items:
        lines.append(f"- Glossary: {', '.join(glossary_items)}")
    return "\n".join(lines) + "\n\n"


def build_asr_hotword_list(context, fallback_title="", limit=12):
    normalized = normalize_series_context(context, fallback_title=fallback_title)
    items = []
    canonical_title = str(normalized.get("canonical_title_en") or "").strip()
    canonical_title_tokens = {
        token.casefold()
        for token in re.split(r"[\s\-_/]+", canonical_title)
        if token
    }

    def is_asr_safe_term(value, require_title_overlap=False):
        text = str(value or "").strip()
        if not text:
            return False
        if len(text) > 40:
            return False
        if re.search(r"[^\w\s'\-一-龯ぁ-ゟ゠-ヿー]", text):
            return False
        if require_title_overlap and canonical_title_tokens:
            lowered_tokens = {
                token.casefold()
                for token in re.split(r"[\s\-_/]+", text)
                if token
            }
            overlap_count = len(lowered_tokens & canonical_title_tokens)
            if overlap_count < 2 and not re.search(r"[一-龯ぁ-ゟ゠-ヿー]", text):
                return False
        return True

    def add_item(value):
        text = str(value or "").strip()
        if text:
            items.append(text)

    add_item(normalized.get("canonical_title_en"))
    add_item(normalized.get("canonical_title_ja"))
    for alias in normalized.get("aliases", []):
        if is_asr_safe_term(alias, require_title_overlap=True):
            add_item(alias)
    for character in normalized.get("characters", [])[:3]:
        if is_asr_safe_term(character.get("name_en")):
            add_item(character.get("name_en"))
        if is_asr_safe_term(character.get("name_ja")):
            add_item(character.get("name_ja"))
    for glossary in normalized.get("glossary", [])[:3]:
        if is_asr_safe_term(glossary.get("preferred_en")):
            add_item(glossary.get("preferred_en"))
        if is_asr_safe_term(glossary.get("source_ja")):
            add_item(glossary.get("source_ja"))

    deduped = _clean_string_list(items)
    return deduped[:limit]
