"""Fetch a tiny CC0/public-domain image dataset for the few-shot demo.

Positives = squirrels. Negatives = "background": empty garden/lawn scenes, plus
a few hard negatives (cats, birds) so the classifier learns "squirrel" rather
than just "an animal exists". All images are CC0 / public-domain via Openverse.

Run:  uv run --group detector python -m tools.fetch_dataset
"""

import os

import httpx

_HDR = {"User-Agent": "save-the-hibiscus/0.1 (learning project)"}
_DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def fetch(query: str, out_dir: str, n: int, start: int = 0) -> int:
    os.makedirs(out_dir, exist_ok=True)
    r = httpx.get(
        "https://api.openverse.org/v1/images/",
        params={"q": query, "license": "cc0,pdm", "page_size": n},
        headers=_HDR,
        timeout=30,
    )
    saved = 0
    for res in r.json().get("results", []):
        try:
            img = httpx.get(res["url"], headers=_HDR, timeout=30, follow_redirects=True).content
            if len(img) < 5000:  # skip tiny/broken
                continue
            with open(os.path.join(out_dir, f"{start + saved:02d}.jpg"), "wb") as f:
                f.write(img)
            saved += 1
        except Exception as e:
            print(f"  skip ({e})")
    return saved


def main() -> None:
    pos = os.path.join(_DATA, "squirrel")
    neg = os.path.join(_DATA, "background")

    print("fetching positives (squirrels)...")
    n_pos = fetch("squirrel", pos, 14)

    print("fetching negatives (empty scenes + hard negatives)...")
    n_neg = 0
    for i, q in enumerate(
        ["empty garden lawn", "backyard fence", "domestic cat outdoors", "bird garden"]
    ):
        n_neg += fetch(q, neg, 5, start=n_neg)

    print(f"\ndone: {n_pos} positives -> {pos}\n      {n_neg} negatives -> {neg}")


if __name__ == "__main__":
    main()
