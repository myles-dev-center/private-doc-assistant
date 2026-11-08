from docassist.names import match

NAMES = {
    "jenner.pdf": ["Bruce Jenner, Individual", "The Right Solution, a Nevada Corporation"],
    "north.pdf": ["Andy North", "Golfers Incorporated"],
    "acme.pdf": ["Acme Widgets GmbH", "Northwind Traders Ltd."],
    "ap.pdf": ["AP-SA-2024-017", "Harbour Metal Works Ltd"],
}


def test_full_names_and_names_without_legal_form():
    assert match("Which law governs the Endorsement Agreement between Bruce Jenner and The Right Solution?", NAMES) == {
        "jenner.pdf": "full"
    }
    assert match("When does the agreement with Golfers Incorporated expire?", NAMES) == {"north.pdf": "full"}
    assert match("When does the agreement with Golfers expire?", NAMES) == {"north.pdf": "full"}
    assert match("Payment terms with Northwind Traders?", NAMES) == {"acme.pdf": "full"}
    assert match("When does contract AP-SA-2024-017 end?", NAMES) == {"ap.pdf": "full"}


def test_short_names_are_partial():
    assert match("What does the Jenner deal say about termination?", NAMES) == {"jenner.pdf": "partial"}
    assert match("Is there an audit clause in the Acme contract?", NAMES) == {"acme.pdf": "partial"}


def test_words_that_are_not_names_do_not_match():
    # "Nevada" is in "a Nevada Corporation", which describes the party, not its name.
    assert match("Which agreements are governed by the laws of Nevada?", NAMES) == {}
    assert match("Which agreements contain a non-compete clause?", NAMES) == {}
    assert match("List all agreements governed by the laws of North Carolina.", NAMES) == {}


def test_a_word_shared_by_two_documents_is_ambiguous():
    names = {"a.pdf": ["Acme Widgets GmbH"], "b.pdf": ["Acme Tools Inc."]}
    assert match("What does the Acme contract say?", names) == {}
    assert match("What does the Acme Tools contract say?", names) == {"b.pdf": "full"}
