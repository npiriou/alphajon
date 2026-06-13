import argparse
from collections import Counter, defaultdict

from objets import (
    ITEM_HOOK_OPTIONAL_ACTIVATION,
    SANS_HOOK_OBJET,
    _HOOKS_OBJET,
    classify_item_hook,
    objets_disponibles,
)


def iter_item_hook_rows():
    seen_classes = set()
    for item in objets_disponibles:
        cls = type(item)
        if cls in seen_classes:
            continue
        seen_classes.add(cls)
        for hook in _HOOKS_OBJET:
            if cls in SANS_HOOK_OBJET[hook]:
                continue
            classification = classify_item_hook(cls, hook)
            yield {
                "item": item.nom,
                "class": cls.__name__,
                "hook": hook,
                "classification": classification,
                "policy_routed": classification == ITEM_HOOK_OPTIONAL_ACTIVATION,
            }


def summarize(rows):
    by_hook = defaultdict(Counter)
    for row in rows:
        by_hook[row["hook"]][row["classification"]] += 1
    return by_hook


def print_markdown(rows):
    rows = list(rows)
    by_hook = summarize(rows)
    total_items = len({row["class"] for row in rows})
    print("# Item Hook Coverage")
    print()
    print(f"Items with at least one implemented hook: {total_items}")
    print(f"Implemented item hooks: {len(rows)}")
    print()
    print("## Summary")
    print()
    print("| Hook | Total | Classification counts | Policy routed |")
    print("| --- | ---: | --- | ---: |")
    for hook in sorted(by_hook):
        counts = by_hook[hook]
        total = sum(counts.values())
        routed = counts.get(ITEM_HOOK_OPTIONAL_ACTIVATION, 0)
        count_text = ", ".join(f"{name}: {count}" for name, count in sorted(counts.items()))
        print(f"| {hook} | {total} | {count_text} | {routed} |")
    print()
    print("## Details")
    print()
    print("| Item | Class | Hook | Classification | Policy routed |")
    print("| --- | --- | --- | --- | --- |")
    for row in sorted(rows, key=lambda r: (r["hook"], r["item"], r["class"])):
        routed = "yes" if row["policy_routed"] else "no"
        print(
            f"| {row['item']} | {row['class']} | {row['hook']} | "
            f"{row['classification']} | {routed} |"
        )


def main():
    parser = argparse.ArgumentParser(description="Report Stage 4 item hook coverage.")
    parser.add_argument(
        "--format",
        choices=("markdown",),
        default="markdown",
        help="Output format.",
    )
    args = parser.parse_args()
    rows = list(iter_item_hook_rows())
    if args.format == "markdown":
        print_markdown(rows)


if __name__ == "__main__":
    main()
