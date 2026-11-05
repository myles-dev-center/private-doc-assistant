from docassist.chunking import approx_tokens, chunk_doc
from docassist.parsing import Page, ParsedDoc


def test_chunks_respect_page_boundaries_and_size():
    sentence = "The supplier shall deliver the goods on time and in full. "
    doc = ParsedDoc("x.pdf", [Page(1, sentence * 30), Page(2, "Short second page.")])
    chunks = chunk_doc(doc, max_tokens=60, overlap=15)
    assert {c.page for c in chunks} == {1, 2}
    assert chunks[-1].text == "Short second page."
    assert all(approx_tokens(c.text) <= 60 + 15 for c in chunks)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_overlap_repeats_tail_sentence():
    doc = ParsedDoc("x.pdf", [Page(1, " ".join(f"Sentence number {i} is here." for i in range(20)))])
    chunks = chunk_doc(doc, max_tokens=30, overlap=10)
    assert len(chunks) > 1
    last_sentence_of_first = chunks[0].text.split(". ")[-1]
    assert last_sentence_of_first.rstrip(".") in chunks[1].text


def test_empty_page_produces_no_chunks():
    assert chunk_doc(ParsedDoc("x.pdf", [Page(1, "   ")])) == []
