import argparse
import json
import os
import sys

from pipeline_profiles import DEFAULT_CONTEXT_MODEL
from series_context import resolve_series_context, save_series_context_file


def main(targets, context_model=DEFAULT_CONTEXT_MODEL, force_refresh=False, save_local=False):
    success = True
    for target in targets:
        try:
            print(json.dumps({"status": "resolving_series_context", "target": target}), file=sys.stderr)
            context, source, cache_path = resolve_series_context(
                target,
                allow_fetch=True,
                enrich_model=context_model,
                force_refresh=force_refresh,
            )
            print(
                json.dumps(
                    {
                        "status": "saved_series_context",
                        "target": target,
                        "source": source,
                        "cache_path": cache_path,
                        "title": context.get("canonical_title_en") or context.get("canonical_title_ja"),
                    }
                ),
                file=sys.stderr,
            )

            if save_local and os.path.exists(target):
                local_path = os.path.join(os.path.dirname(target), "series_context.json")
                save_series_context_file(context, local_path)
                print(json.dumps({"status": "saved_local_series_context", "target": target, "file": local_path}), file=sys.stderr)
        except Exception as exc:
            success = False
            print(json.dumps({"error": f"{target}: {exc}"}), file=sys.stderr)
    if not success:
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch, enrich, and cache reusable series context for future YakuZen runs.")
    parser.add_argument("targets", nargs="+", help="One or more video paths, JSON paths, or raw series titles.")
    parser.add_argument(
        "--context-model",
        default=DEFAULT_CONTEXT_MODEL,
        help="Ollama model used to normalize fetched metadata into structured series context.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore existing global cache entries and fetch/enrich again.",
    )
    parser.add_argument(
        "--save-local",
        action="store_true",
        help="Also save a series_context.json file next to each file target.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        args.targets,
        context_model=args.context_model,
        force_refresh=args.force_refresh,
        save_local=args.save_local,
    )
