from docassist.evaluation import dates_in, docs_in, ids_in, numbers_in, score, summarize
from docassist.pipeline import NOT_FOUND, Answer, Source


def q(kind, value=None, qtype="lookup", source="a.pdf"):
    return {"id": "Q1", "type": qtype, "tag": "t", "question": "?", "expected": {"kind": kind, "value": value}, "source": source}


def test_date_formats_normalized():
    assert dates_in("ends on 31 March 2027") == {"2027-03-31"}
    assert dates_in("ends March 31, 2027") == {"2027-03-31"}
    assert dates_in("ends 31.03.2027 or 2027-04-01") == {"2027-03-31", "2027-04-01"}
    assert dates_in("ends 3/31/2027") == {"2027-03-31"}
    assert dates_in("made this 31st day of March, 2027") == {"2027-03-31"}


def test_numbers_normalized():
    assert 2_500_000 in numbers_in("capped at EUR 2,500,000.00")
    assert 2_500_000 in numbers_in("2.5 million euros")
    assert 45 in numbers_in("net 45 days")


def test_lookup_correct_with_citation():
    ans = Answer("?", "It ends on 28 February 2027 [1].", "search", sources=[Source(1, "a.pdf", 1, "")])
    s = score(q("date", "2027-02-28"), ans)
    assert s.correct and s.citation_ok and not s.hallucinated


def test_wrong_answer_is_hallucination_refusal_is_not():
    wrong = score(q("date", "2027-02-28"), Answer("?", "It ends on 28 February 2026 [1].", "search"))
    assert not wrong.correct and wrong.hallucinated
    miss = score(q("date", "2027-02-28"), Answer("?", NOT_FOUND, "not_found"))
    assert not miss.correct and not miss.hallucinated and miss.refused


def test_unanswerable():
    assert score(q("refusal", qtype="unanswerable"), Answer("?", NOT_FOUND, "not_found")).correct
    s = score(q("refusal", qtype="unanswerable"), Answer("?", "The cap is EUR 1,000,000.", "search"))
    assert s.hallucinated and not s.correct


def test_aggregate_requires_exact_set_and_reads_sql_rows():
    exp = ["AP-SA-2024-017", "AP-LIC-2023-020"]
    ans = Answer("?", "Two contracts match.", "sql", columns=["contract_id"], rows=[["AP-SA-2024-017"], ["AP-LIC-2023-020"]])
    assert score(q("ids", exp, "aggregate"), ans).correct
    extra = Answer("?", "AP-SA-2024-017, AP-LIC-2023-020, AP-SA-2024-071", "search")
    s = score(q("ids", exp, "aggregate"), extra)
    assert not s.correct and s.recall == 1.0 and s.detail["extra"] == ["AP-SA-2024-071"]
    assert ids_in("AP-SA-2024-017 and AP-NDA-2025-016") == {"AP-SA-2024-017", "AP-NDA-2025-016"}


def test_aggregate_by_document_aliases():
    docs = {"a.pdf": ["a.pdf", "Freeze Tag"], "b.pdf": ["b.pdf", "Corio"], "c.pdf": ["c.pdf", "Adams Golf"]}
    assert docs_in("FREEZE TAG, INC. and corio", docs) == {"a.pdf", "b.pdf"}
    assert docs_in("Corioland", docs) == set()
    ans = Answer("?", "Two match.", "sql", columns=["filename"], rows=[["a.pdf"], ["c.pdf"]])
    assert score(q("docs", ["a.pdf", "c.pdf"], "aggregate"), ans, docs).correct
    assert not score(q("docs", ["a.pdf"], "aggregate"), ans, docs).correct


def test_summary():
    scores = [
        score(q("date", "2027-02-28"), Answer("?", "28 February 2027", "search", sources=[Source(1, "a.pdf", 1, "")])),
        score(q("refusal", qtype="unanswerable"), Answer("?", "Yes it is 5.", "search")),
    ]
    s = summarize(scores)
    assert s["accuracy"] == 0.5 and s["hallucination_rate"] == 0.5 and s["citation_accuracy"] == 1.0
