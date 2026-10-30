"""Comparing two products ("difference between paracetamol and ibuprofen",
"Paracetamol 500mg vs Ibuprofen 200mg", "compare the first two", "which is
better, Tylenol or Advil"). The model answers these in its own words, and the
knowledge base has no comparison text. (On Oct 22 I called one phrase it used —
"gentler on the stomach" — an improvisation; the paracetamol document actually
says exactly that. Judging from memory is the problem, so this module does not
judge: it quotes.) The reply puts each product's own catalog entry and its own
Overview, Dosage and Warnings text side by side and says plainly that it will
not call one "better", because that depends on the person. Dependency-light: the
catalog is passed in; the product documents are read from the knowledge base.
"""

import re

from app import brands, data_ingest, interactions

_COMPARE_RE = re.compile(
    r"\b(?:differences?\s+between|difference\s+of|differ|vs\.?|versus|compare[ds]?|comparison|"
    r"which\s+(?:one\s+)?(?:is|are)\s+(?:better|stronger|safer|best|worse|gentler|cheaper)|"
    r"(?:is|are)\s+.{1,50}\s+(?:better|stronger|safer|gentler)\s+than|better\s+than|instead\s+of)\b",
    re.IGNORECASE,
)
_NOT_A_COMPARISON_RE = re.compile(r"\b(?:together|combine[ds]?|at\s+the\s+same\s+time|order|buy|cancel|price|cost|how\s+much)\b", re.IGNORECASE)
_MAX_WORDS = 26
_MAX_PRODUCTS = 3
_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "last": -1}


def is_comparison_request(text: str) -> bool:
    words = re.findall(r"[a-z']+", text.lower())
    if not words or len(words) > _MAX_WORDS or "@" in text:
        return False
    return bool(_COMPARE_RE.search(text)) and not _NOT_A_COMPARISON_RE.search(text)


def _section(source: str, heading: str) -> str:
    path = data_ingest.KB_DIR / source
    if not path.is_file():
        return ""
    match = re.search(rf"^##\s+{re.escape(heading)}\s*\n(.*?)(?=^##\s|\Z)", path.read_text(encoding="utf-8"), re.MULTILINE | re.DOTALL)
    return " ".join(match.group(1).split()) if match else ""


def _representative(ingredient: str, catalog: list[dict]) -> dict | None:
    """The plain single-ingredient adult product for an ingredient word."""
    options = [p for p in catalog if p["active_ingredient"].lower() == ingredient.lower()]
    adults = [p for p in options if "children" not in p["name"].lower()]
    return (adults or options or [None])[0]


def targets(text: str, catalog: list[dict], shown: list[dict] | None) -> tuple[list[dict], list[str]] | None:
    """(products to compare, notes) or None when two products cannot be told."""
    products: list[dict] = []
    notes: list[str] = []
    for label, _classes, product_id in interactions.mentions(text, catalog):
        product = next((p for p in catalog if p["id"] == product_id), None) if product_id else None
        if product is None:
            brand = label.split(" (")[0].lower()
            ingredient = brands.BRAND_INGREDIENT.get(brand) or label.split(" (")[0]
            if brand in brands.NOT_STOCKED:
                return None  # a medicine with nothing to compare against
            product = _representative(ingredient, catalog)
            if product is None:
                continue
            if brand in brands.BRAND_INGREDIENT and not brands.is_ingredient_itself(brand):
                notes.append(f"We don't carry {brand.title()}; its active ingredient is {ingredient.lower()}, which we carry as {product['name']}.")
            elif product["name"].lower() not in text.lower() and sum(1 for p in catalog if ingredient.lower() in p["active_ingredient"].lower().split(" + ")) > 1:
                notes.append(f"For {ingredient.lower()} I've used {product['name']}.")
        if product not in products:
            products.append(product)
    if len(products) >= 2:
        return products[:_MAX_PRODUCTS], notes
    if shown and len(shown) >= 2:
        words = re.findall(r"[a-z0-9']+", text.lower())
        positions = [(_ORDINALS[w] if _ORDINALS[w] > 0 else len(shown)) for w in words if w in _ORDINALS]
        positions += [int(n) for n in re.findall(r"(?:number|no\.?|option|item|#)\s*(\d{1,2})", text, re.IGNORECASE)]
        if re.search(r"\bfirst\s+two\b", text, re.IGNORECASE):
            positions = [1, 2]
        elif re.search(r"\blast\s+two\b", text, re.IGNORECASE):
            positions = [len(shown) - 1, len(shown)]
        elif re.search(r"\b(?:both|these\s+two|those\s+two)\b", text, re.IGNORECASE) and len(shown) == 2:
            positions = [1, 2]
        positions = list(dict.fromkeys(positions))
        if len(positions) >= 2 and all(1 <= n <= len(shown) for n in positions):
            return [shown[n - 1] for n in positions[:_MAX_PRODUCTS]], notes
    return None


def comparison_reply(products: list[dict], notes: list[str], disclaimer: str) -> str:
    blocks = []
    for product in products:
        source = f"{product['id']}.md"
        lines = [f"{product['name']} ({product['id']}) — ${product['price_usd']:.2f}", f"  Active ingredient: {product['active_ingredient']}"]
        for label, heading in (("What it is", "Overview"), ("Dosage", "Dosage"), ("Warnings", "Warnings")):
            text = _section(source, heading)
            if text:
                lines.append(f"  {label}: {text}")
        blocks.append("\n".join(lines))
    head = "Here's how they compare, taken straight from our product information:"
    note = ("\n\n" + " ".join(notes)) if notes else ""
    return (
        f"{head}{note}\n\n" + "\n\n".join(blocks)
        + "\n\nI can't say one is \"better\" — it depends on you: your age, health conditions and other medicines. "
        "Tell me about any conditions and I'll check these against them, or a pharmacist can advise.\n\n" + disclaimer
    )
