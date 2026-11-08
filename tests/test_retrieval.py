import pytest

from docassist.retrieval import fts_query, rrf
from docassist.store import Store


def test_fts_query_keeps_identifiers_as_phrases():
    q = fts_query("When does contract AP-SA-2024-017 end?")
    assert '"AP SA 2024 017"' in q
    assert '"contract"' in q and '"When"' not in q and '"does"' not in q


def test_fts_query_escapes_quotes_and_operators():
    q = fts_query('What about "NEAR" OR AND (x)?')
    assert q.count('"') % 2 == 0


def test_rrf_rewards_agreement():
    assert rrf([[1, 2, 3], [3, 1, 4]])[0] == 1
    assert set(rrf([[1], [2]])) == {1, 2}


@pytest.mark.parametrize("question", ['he said "hi"', "NOT", "a*b", "x:y", "^start"])
def test_fts_query_never_breaks_sqlite(tmp_path, question):
    store = Store(tmp_path / "t.db")
    store.fts_search(fts_query(question), 5)  # must not raise


def test_mirrored_ids_are_distinguished(tmp_path):
    import numpy as np

    store = Store(tmp_path / "t.db")
    for cid in ("AP-SA-2024-017", "AP-SA-2024-071"):
        doc_id = store.add_document(f"{cid}.pdf", "sha", 1, 0, None)
        store.add_chunks(doc_id, [(0, 1, f"Contract No.: {cid}", f"Contract No.: {cid}")], np.ones((1, 4), np.float32))
    store.commit()
    hits = store.get_chunks(store.fts_search(fts_query("contract AP-SA-2024-071"), 2))
    assert hits[0].filename == "AP-SA-2024-071.pdf"


def test_old_library_gains_model_columns(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE documents (id INTEGER PRIMARY KEY, filename TEXT UNIQUE NOT NULL, sha256 TEXT NOT NULL, "
        "pages INTEGER, ocr_pages INTEGER DEFAULT 0, header TEXT, ingested_at TEXT DEFAULT CURRENT_TIMESTAMP)"
    )
    db.execute("INSERT INTO documents(filename, sha256) VALUES ('a.pdf', 'x')")
    db.commit()
    db.close()
    (doc,) = Store(path).documents()
    assert doc["filename"] == "a.pdf" and doc["embed_model"] is None and doc["extract_model"] is None


def test_chinese_text_is_searchable_by_keyword(tmp_path):
    import numpy as np

    from docassist.retrieval import fts_text

    store = Store(tmp_path / "t.db")
    texts = {"a.pdf": "第五条 付款条件：买方应在收到发票后三十日内付款。", "b.pdf": "第九条 适用法律：本合同适用中华人民共和国法律。"}
    for name, text in texts.items():
        doc_id = store.add_document(name, "sha", 1, 0, None)
        store.add_chunks(doc_id, [(0, 1, text, fts_text(text))], np.ones((1, 4), np.float32))
    store.commit()
    hits = store.get_chunks(store.fts_search(fts_query("付款条件是什么？"), 2))
    assert hits[0].filename == "a.pdf"
    assert fts_text("Payment within 30 days.") == "Payment within 30 days."  # non-CJK text is indexed as before
