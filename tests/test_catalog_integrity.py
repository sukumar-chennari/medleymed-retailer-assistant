"""Data-integrity checks across catalog.json, the knowledge_base/*.md
files, and CLARIFYING_QUESTIONS' hardcoded product_ids.

None of these would raise a Python exception anywhere if broken — a typo'd
product_id in a clarifying-question branch just silently renders an empty
product list (_render_products filters out None results from
store.find_product); an orphaned or missing knowledge_base file just
silently under/over-covers the catalog. These tests exist to catch that
class of silent data drift the moment a catalog or knowledge-base edit
introduces it, rather than waiting for a chat reply to look oddly wrong.
"""

from app import agent, data_ingest, store

VALID_CATEGORIES = {"fever", "cold"}
REQUIRED_PRODUCT_FIELDS = {"id", "name", "category", "active_ingredient", "price_usd", "description"}


class TestCatalogSchema:
    def test_every_product_has_the_required_fields(self):
        for product in store.get_catalog():
            missing = REQUIRED_PRODUCT_FIELDS - product.keys()
            assert not missing, f"{product.get('id')} is missing fields: {missing}"

    def test_every_product_has_non_empty_field_values(self):
        for product in store.get_catalog():
            for field in REQUIRED_PRODUCT_FIELDS:
                assert product[field] not in (None, ""), f"{product['id']}.{field} is empty"

    def test_every_product_category_is_one_the_app_understands(self):
        for product in store.get_catalog():
            assert product["category"] in VALID_CATEGORIES, product["id"]

    def test_product_ids_are_unique(self):
        ids = [p["id"] for p in store.get_catalog()]
        assert len(ids) == len(set(ids))

    def test_prices_are_positive(self):
        for product in store.get_catalog():
            assert product["price_usd"] > 0, product["id"]


class TestClarifyingQuestionsReferenceRealProducts:
    def test_every_branch_product_id_exists_in_the_catalog(self):
        for trigger, rule in agent.CLARIFYING_QUESTIONS.items():
            for branch in rule["branches"]:
                for product_id in branch["product_ids"]:
                    assert store.find_product(product_id) is not None, (
                        f"CLARIFYING_QUESTIONS[{trigger!r}] references unknown product_id {product_id!r}"
                    )

    def test_every_branch_product_matches_the_triggers_own_category(self):
        # A cough/fever/cold branch recommending a product from the wrong
        # category would be a silent, easy-to-miss data-entry mistake.
        for trigger, rule in agent.CLARIFYING_QUESTIONS.items():
            expected_category = "fever" if trigger == "fever" else "cold"
            for branch in rule["branches"]:
                for product_id in branch["product_ids"]:
                    product = store.find_product(product_id)
                    assert product["category"] == expected_category, (
                        f"{trigger}/{product_id} has category {product['category']!r}, expected {expected_category!r}"
                    )


class TestKnowledgeBaseCoversTheCatalog:
    def test_every_catalog_product_has_a_knowledge_base_file(self):
        kb_filenames = {name for name, _ in data_ingest.load_documents()}
        for product in store.get_catalog():
            assert f"{product['id']}.md" in kb_filenames, f"missing knowledge_base/{product['id']}.md"

    def test_every_knowledge_base_file_maps_to_a_real_catalog_product(self):
        # The reverse check — an orphaned file for a since-removed product
        # would otherwise sit there unnoticed, still getting ingested.
        catalog_ids = {p["id"] for p in store.get_catalog()}
        for filename, _ in data_ingest.load_documents():
            product_id = filename.removesuffix(".md")
            assert product_id in catalog_ids, f"knowledge_base/{filename} has no matching catalog product"
