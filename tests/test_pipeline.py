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
    write_txt(docs, "a.txt", "Contract No.: AP-SA-2024-017. The agreement ends on 31 January 2027.")
    write_txt(docs, "notes.xyz", "ignored")
    results = assistant.ingest_dir(docs)
    assert [r["status"] for r in results] == ["ingested"]
    rec = assistant.records()[0]
    assert rec["contract_id"] == "AP-SA-2024-017"
    assert rec["auto_renewal"] is True and rec["contract_value"] == 1200.5  # coerced from "yes" / "EUR 1,200.50"
    assert assistant.ingest_dir(docs)[0]["status"] == "unchanged"
    write_txt(docs, "a.txt", "Contract No.: AP-SA-2024-017. Amended: ends on 31 March 2027.")
    assert assistant.ingest_dir(docs)[0]["status"] == "ingested"
    assert len(assistant.store.documents()) == 1


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


def test_empty_sql_result_falls_back_to_search(assistant, tmp_path, fake_llm, monkeypatch):
    assistant.ingest_file(write_txt(tmp_path, "a.txt", "Contract No.: AP-SA-2024-017."))
    import docassist.pipeline as pl

    monkeypatch.setattr(pl, "route", lambda *a: {"route": "sql", "sql": "SELECT contract_id FROM contracts WHERE 0"})
    assert assistant.ask("List all contracts that end in 2027?").route != "sql"


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
    monkeypatch.setattr(pl, "route", lambda *a: {"route": "sql", "sql": sql})
    assert assistant.ask("List all contracts and their liability caps?").route != "sql"


def test_question_naming_a_document_is_answered_from_its_text(assistant, tmp_path, monkeypatch):
    assistant.ingest_file(write_txt(tmp_path, "a.txt", "Contract No.: AP-SA-2024-017. Payment within 45 days."))
    import docassist.pipeline as pl

    monkeypatch.setattr(pl, "route", lambda *a: {"route": "sql", "sql": "SELECT contract_id FROM contracts"})
    ans = assistant.ask("List the payment terms of contract AP-SA-2024-017?")
    assert ans.route == "document" and ans.sources[0].filename == "a.txt"


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
