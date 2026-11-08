"""Pipeline tests with a fake LLM: ingestion, dedup, routing, citations, refusals."""

from pathlib import Path

from docx import Document


def write_txt(folder: Path, name: str, text: str) -> Path:
    p = folder / name
    p.write_text(text, encoding="utf-8")
    return p


def test_ingest_extract_and_skip_unchanged(assistant, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    write_txt(docs, "a.txt", "Contract No.: AP-SA-2024-017. The agreement ends on 31 January 2027. It renews automatically.")
    write_txt(docs, "notes.xyz", "ignored")
    results = assistant.ingest_dir(docs)
    assert [r["status"] for r in results] == ["ingested"]
    rec = assistant.records()[0]
    assert rec["contract_id"] == "AP-SA-2024-017"
    assert rec["auto_renewal"] is True and rec["contract_value"] == 1200.5  # coerced from "yes" / "EUR 1,200.50"
    ev = assistant.store.evidence()
    assert ev[("a.txt", "auto_renewal")] == {"quote": "It renews automatically", "page": None, "verified": True}
    doc = assistant.store.documents()[0]
    assert (doc["embed_model"], doc["extract_model"]) == (assistant.s.embed_model, assistant.s.llm_model)
    assert doc["ingest_seconds"] is not None and doc["ingested_at"]
    assert assistant.ingest_dir(docs)[0]["status"] == "unchanged"
    write_txt(docs, "a.txt", "Contract No.: AP-SA-2024-017. Amended: ends on 31 March 2027.")
    assert assistant.ingest_dir(docs)[0]["status"] == "ingested"
    assert len(assistant.store.documents()) == 1


def test_clause_without_a_quote_in_the_text_is_not_trusted(assistant, tmp_path):
    # The model says "renews automatically" but its quote is nowhere in the document.
    assistant.ingest_file(write_txt(tmp_path, "a.txt", "Contract No.: AP-SA-2024-017. Ends on 31 January 2027."))
    rec = assistant.records()[0]
    assert rec["auto_renewal"] is None and rec["contract_value"] == 1200.5
    assert assistant.store.evidence()[("a.txt", "auto_renewal")]["verified"] is False
    assistant.s = assistant.s.model_copy(update={"require_clause_quote": False})
    assistant.ingest_file(write_txt(tmp_path, "b.txt", "Contract No.: AP-SA-2024-071."))
    assert assistant.records()[1]["auto_renewal"] is True


def test_context_header_is_indexed(assistant, tmp_path):
    p = write_txt(tmp_path, "a.txt", "The Supplier shall deliver.")
    assistant.ingest_file(p)
    row = assistant.store.db.execute("SELECT search_text FROM chunks").fetchone()
    assert row[0].startswith("Supply Agreement | Test Co") or "Test Co" in row[0]


def test_docx_tables_are_parsed(assistant, tmp_path):
    d = Document()
    d.add_paragraph("Contract No.: AP-LIC-2023-020")
    t = d.add_table(rows=1, cols=2)
    t.rows[0].cells[0].text, t.rows[0].cells[1].text = "Total", "90,000.00"
    d.save(tmp_path / "c.docx")
    assistant.ingest_file(tmp_path / "c.docx")
    text = assistant.store.db.execute("SELECT text FROM chunks").fetchone()[0]
    assert "90,000.00" in text


def test_search_answer_has_citation(assistant, tmp_path):
    assistant.ingest_file(write_txt(tmp_path, "a.txt", "Contract No.: AP-SA-2024-017. Payment within 45 days."))
    ans = assistant.ask("What are the payment terms?")
    assert ans.route == "search"
    assert ans.sources and ans.sources[0].filename == "a.txt"


def test_not_found(assistant, tmp_path):
    assistant.ingest_file(write_txt(tmp_path, "a.txt", "Contract No.: AP-SA-2024-017."))
    ans = assistant.ask("What does Zorblax cost?")
    assert ans.route == "not_found" and ans.sources == []


def test_aggregate_routes_to_sql(assistant, tmp_path):
    assistant.ingest_file(write_txt(tmp_path, "a.txt", "Contract No.: AP-SA-2024-017."))
    ans = assistant.ask("List all contracts that end in 2027?")
    assert ans.route == "sql"
    assert ans.rows == [["AP-SA-2024-017", "2027-01-31"]]


def test_structured_off_never_uses_sql(assistant, tmp_path):
    assistant.s = assistant.s.model_copy(update={"structured": False})
    assistant.ingest_file(write_txt(tmp_path, "a.txt", "Contract No.: AP-SA-2024-017."))
    assert assistant.ask("List all contracts that end in 2027?").route != "sql"


def test_scope_restricts_search_to_named_document(assistant, tmp_path):
    for cid in ("AP-SA-2024-017", "AP-SA-2024-071"):
        assistant.ingest_file(write_txt(tmp_path, f"{cid}.txt", f"Contract No.: {cid}. Payment within 45 days."))
    scoped = assistant.scope("What are the payment terms in contract AP-SA-2024-071?")
    names = {v: k for k, v in assistant.store.doc_ids_by_filename().items()}
    assert [names[i] for i in scoped] == ["AP-SA-2024-071.txt"]
    hits = assistant.retriever.search("payment terms", 5, scoped)
    assert {h.filename for h in hits} == {"AP-SA-2024-071.txt"}
    assert assistant.scope("Which contracts renew automatically?") is None


def test_empty_sql_result_is_an_answer(assistant, tmp_path, monkeypatch):
    # No rows is a real answer ("none match"), not a reason to let search guess one.
    assistant.ingest_file(write_txt(tmp_path, "a.txt", "Contract No.: AP-SA-2024-017."))
    import docassist.pipeline as pl

    sql = "SELECT filename, contract_id FROM contracts WHERE governing_law LIKE '%Narnia%'"
    monkeypatch.setattr(pl, "route", lambda *a: {"route": "sql", "sql": sql, "document": None})
    monkeypatch.setattr(pl, "relax_sql", lambda *a: None)
    ans = assistant.ask("List all contracts governed by the laws of Narnia?")
    assert ans.route == "sql" and ans.rows == [] and ans.answer == pl.NO_MATCH


def test_empty_sql_result_is_relaxed_once(assistant, tmp_path, monkeypatch):
    assistant.ingest_file(write_txt(tmp_path, "a.txt", "Contract No.: AP-SA-2024-017."))
    import docassist.pipeline as pl

    strict = "SELECT filename, contract_id FROM contracts WHERE counterparty = 'test co'"
    loose = "SELECT filename, contract_id FROM contracts WHERE counterparty LIKE '%test co%'"
    seen = []

    def relax(llm, schema, db, question, sql):
        seen.append(sql)
        return loose

    monkeypatch.setattr(pl, "route", lambda *a: {"route": "sql", "sql": strict, "document": None})
    monkeypatch.setattr(pl, "relax_sql", relax)
    ans = assistant.ask("List all contracts with Test Co?")
    assert seen == [strict] and ans.sql == loose and ans.rows == [["a.txt", "AP-SA-2024-017"]]


def test_table_answer_carries_evidence(assistant, tmp_path, monkeypatch):
    assistant.ingest_file(write_txt(tmp_path, "a.txt", "Contract No.: AP-SA-2024-017. It renews automatically."))
    import docassist.pipeline as pl

    sql = "SELECT contract_id FROM contracts WHERE auto_renewal = 1"
    monkeypatch.setattr(pl, "route", lambda *a: {"route": "sql", "sql": sql, "document": None})
    ans = assistant.ask("Which contracts renew automatically?")
    assert [(e.filename, e.field, e.quote, e.verified) for e in ans.evidence] == [
        ("a.txt", "auto_renewal", "It renews automatically", True)
    ]


def test_broken_router_reply_falls_back_to_search(assistant, tmp_path, monkeypatch):
    assistant.ingest_file(write_txt(tmp_path, "a.txt", "Contract No.: AP-SA-2024-017. Payment within 45 days."))
    import docassist.pipeline as pl

    def broken(*a):
        raise ValueError("Unterminated string starting at: line 1 column 25")

    monkeypatch.setattr(pl, "route", broken)
    ans = assistant.ask("List all contracts that end in 2027?")
    assert ans.route == "search" and ans.sources


def test_sql_rows_without_values_fall_back_to_search(assistant, tmp_path, monkeypatch):
    assistant.ingest_file(write_txt(tmp_path, "a.txt", "Contract No.: AP-SA-2024-017."))
    import docassist.pipeline as pl

    sql = "SELECT contract_id, liability_cap FROM contracts"
    monkeypatch.setattr(pl, "route", lambda *a: {"route": "sql", "sql": sql, "document": None})
    assert assistant.ask("List all contracts and their liability caps?").route != "sql"


def test_question_naming_a_document_is_answered_from_its_text(assistant, tmp_path, monkeypatch):
    assistant.ingest_file(write_txt(tmp_path, "a.txt", "Contract No.: AP-SA-2024-017. Payment within 45 days."))
    import docassist.pipeline as pl

    calls = []
    monkeypatch.setattr(pl, "route", lambda *a: calls.append(a) or {"route": "sql", "sql": "SELECT contract_id FROM contracts"})
    ans = assistant.ask("What are the payment terms of contract AP-SA-2024-017?")
    assert ans.route == "document" and ans.sources[0].filename == "a.txt"
    assert calls == []  # a full name is certain enough: no routing call


def test_named_document_in_a_question_about_many_goes_to_the_router(assistant, tmp_path, monkeypatch):
    assistant.ingest_file(write_txt(tmp_path, "a.txt", "Contract No.: AP-SA-2024-017. Payment within 45 days."))
    import docassist.pipeline as pl

    seen = []
    sql = "SELECT filename, contract_id FROM contracts WHERE counterparty LIKE '%Test Co%'"

    def router(llm, schema, question, today, candidates=None):
        seen.append([c["filename"] for c in candidates or []])
        return {"route": "sql", "sql": sql, "document": None}

    monkeypatch.setattr(pl, "route", router)
    ans = assistant.ask("Which other contracts do we have with Test Co besides AP-SA-2024-017?")
    assert seen == [["a.txt"]] and ans.route == "sql"


def test_short_name_is_confirmed_by_the_router(assistant, tmp_path, monkeypatch):
    assistant.ingest_file(write_txt(tmp_path, "a.txt", "Contract No.: AP-SA-2024-017. Payment within 45 days."))
    import docassist.pipeline as pl

    monkeypatch.setattr(assistant, "named_documents", lambda q: {"a.txt": "partial"})
    monkeypatch.setattr(pl, "route", lambda *a: {"route": "document", "sql": None, "document": "a.txt"})
    ans = assistant.ask("What are the payment terms in the Brightwater deal?")
    assert ans.route == "document" and ans.sources[0].filename == "a.txt"


def test_pinned_document(assistant, tmp_path):
    assistant.ingest_file(write_txt(tmp_path, "a.txt", "Contract No.: AP-SA-2024-017. Payment within 45 days."))
    assistant.ingest_file(write_txt(tmp_path, "b.txt", "Contract No.: AP-SA-2024-071. Payment within 90 days."))
    ans = assistant.ask("What are the payment terms?", document="b.txt")
    assert ans.route == "document" and {s.filename for s in ans.sources} == {"b.txt"}
    assert ans.model and ans.usage is not None


def test_library_built_with_another_embedding_model_is_refused(assistant, tmp_path):
    import pytest

    from docassist.pipeline import IndexMismatch

    assistant.ingest_file(write_txt(tmp_path, "a.txt", "Contract No.: AP-SA-2024-017."))
    assistant.s = assistant.s.model_copy(update={"embed_model": "other-embedder"})
    with pytest.raises(IndexMismatch):
        assistant.ask("What are the payment terms?")
    with pytest.raises(IndexMismatch):
        assistant.ingest_file(write_txt(tmp_path, "b.txt", "Contract No.: AP-SA-2024-071."))
    assistant.store.delete_document("a.txt")  # an empty library can be rebuilt with any model
    assert assistant.ingest_file(tmp_path / "b.txt")["status"] == "ingested"


def test_named_document_is_read_whole_with_its_record(assistant, tmp_path, fake_llm):
    assistant.ingest_file(write_txt(tmp_path, "a.txt", "Contract No.: AP-SA-2024-017. Payment within 45 days."))
    assistant.ingest_file(write_txt(tmp_path, "b.txt", "Contract No.: AP-SA-2024-071. Payment within 90 days."))
    fake_llm.calls.clear()
    ans = assistant.ask("What are the payment terms in contract AP-SA-2024-071?")
    prompt = fake_llm.calls[-1][1]
    assert ans.route == "document" and {s.filename for s in ans.sources} == {"b.txt"}
    assert "90 days" in prompt and "45 days" not in prompt and "Fields extracted" in prompt


def test_long_named_document_falls_back_to_search(assistant, tmp_path):
    assistant.s = assistant.s.model_copy(update={"full_doc_chars": 10})
    assistant.ingest_file(write_txt(tmp_path, "a.txt", "Contract No.: AP-SA-2024-017. Payment within 45 days."))
    assert assistant.ask("What are the payment terms in contract AP-SA-2024-017?").route == "search"


def test_reprocess_keeps_number_and_edits(assistant, tmp_path):
    assistant.ingest_file(write_txt(tmp_path, "a.txt", "Contract No.: AP-SA-2024-017. It renews automatically."))
    assistant.ingest_file(write_txt(tmp_path, "b.txt", "Contract No.: AP-SA-2024-071."))
    doc = assistant.store.find_document("a.txt")
    assistant.store.edit_value(doc["id"], "counterparty", "Real Name GmbH", "Test Co")
    r = assistant.reprocess("a.txt")
    assert r["status"] == "reprocessed"
    after = assistant.store.find_document("a.txt")
    assert after["id"] == doc["id"]  # same number in the library
    rec = next(x for x in assistant.records() if x["filename"] == "a.txt")
    assert rec["counterparty"] == "Real Name GmbH" and rec["contract_id"] == "AP-SA-2024-017"
    assert [d["filename"] for d in assistant.store.documents()] == ["a.txt", "b.txt"]
    assert assistant.ask("What are the payment terms?", document="a.txt").sources  # chunks were rebuilt


def test_manual_edit_is_used_by_table_questions_and_can_be_undone(assistant, tmp_path, monkeypatch):
    import docassist.pipeline as pl

    assistant.ingest_file(write_txt(tmp_path, "a.txt", "Contract No.: AP-SA-2024-017."))
    doc = assistant.store.find_document("a.txt")
    assistant.store.edit_value(doc["id"], "governing_law", "Germany", None)
    sql = "SELECT filename, contract_id FROM contracts WHERE governing_law LIKE '%Germany%'"
    monkeypatch.setattr(pl, "route", lambda *a: {"route": "sql", "sql": sql, "document": None})
    assert assistant.ask("List all contracts governed by German law?").rows == [["a.txt", "AP-SA-2024-017"]]
    assistant.store.undo_edit(doc["id"], "governing_law")
    assert assistant.records()[0]["governing_law"] is None


def test_reembed_keeps_extraction_and_switches_model(assistant, tmp_path, fake_llm):
    assistant.ingest_file(write_txt(tmp_path, "a.txt", "Contract No.: AP-SA-2024-017. It renews automatically."))
    assistant.ingest_file(write_txt(tmp_path, "b.txt", "Contract No.: AP-SA-2024-071. 付款条件：三十日内付款。"))
    before = {r["filename"]: r for r in assistant.records()}
    fake_llm.calls.clear()
    assistant.s = assistant.s.model_copy(update={"embed_model": "other-embedder"})
    assert assistant.index_problem()  # other model: the library can't be searched yet
    for f in ("a.txt", "b.txt"):
        assert assistant.reembed(f)["status"] == "re-embedded"
    assert fake_llm.calls == []  # no chat call: extraction is kept
    assert {r["filename"]: r for r in assistant.records()} == before
    assert assistant.index_problem() is None
    assert {d["embed_model"] for d in assistant.store.documents()} == {"other-embedder"}
    assert assistant.ask("What are the payment terms?", document="b.txt").sources


def test_unspace_cjk_reverses_fts_text():
    from docassist.retrieval import fts_text, unspace_cjk

    for text in ("Payment within 30 days.", "第五条 付款条件：买方应在收到发票后三十日内付款。", "Header | 合同 AP-1\n条款"):
        assert unspace_cjk(fts_text(text)).split() == text.split()


def test_stopped_ingest_writes_nothing(assistant, tmp_path):
    import pytest

    from docassist.activity import Cancelled

    def stop_at_embed(stage, done=0, total=0, **info):
        if stage == "embed":
            raise Cancelled("test")

    with pytest.raises(Cancelled):
        assistant.ingest_file(write_txt(tmp_path, "a.txt", "Contract No.: AP-SA-2024-017."), progress=stop_at_embed)
    assert assistant.store.documents() == []
