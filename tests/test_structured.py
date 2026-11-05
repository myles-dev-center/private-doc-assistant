import sqlite3
from pathlib import Path

import pytest

from docassist.pipeline import DEFAULT_SCHEMA
from docassist.structured import Schema, coerce, normalize_date, records_db, run_sql


@pytest.fixture
def schema():
    return Schema.load(DEFAULT_SCHEMA)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("31 March 2027", "2027-03-31"),
        ("March 31, 2027", "2027-03-31"),
        ("31.03.2027", "2027-03-31"),
        ("2027-03-31", "2027-03-31"),
        ("soon", None),
    ],
)
def test_normalize_date(raw, expected):
    assert normalize_date(raw) == expected


@pytest.mark.parametrize(
    "value,ftype,expected",
    [
        ("EUR 2,500,000.00", "number", 2_500_000.0),
        ("net 45 days", "number", 45.0),
        (45, "number", 45.0),
        ("null", "number", None),
        ("Yes", "boolean", True),
        (False, "boolean", False),
        ("  Germany ", "text", "Germany"),
        ("N/A", "text", None),
    ],
)
def test_coerce(value, ftype, expected):
    assert coerce(value, ftype) == expected


def test_schema_loads(schema):
    assert schema.name == "contract"
    assert "contract_id" in schema.columns
    assert schema.header_fields[0] == "contract_id"


def test_bad_schema_type_rejected(tmp_path: Path):
    p = tmp_path / "s.yaml"
    p.write_text("name: x\nfields:\n  - {name: a, type: blob, description: d}\n")
    with pytest.raises(ValueError):
        Schema.load(p)


def test_records_db_is_read_only(schema):
    db = records_db(schema, [{"filename": "a.pdf", "contract_id": "AP-SA-2024-017", "end_date": "2027-02-28"}])
    cols, rows = run_sql(db, "SELECT contract_id FROM contracts WHERE end_date LIKE '2027%'")
    assert rows == [["AP-SA-2024-017"]]
    for bad in ["DELETE FROM contracts", "SELECT 1; DROP TABLE contracts", "ATTACH DATABASE 'x' AS y"]:
        with pytest.raises(ValueError):
            run_sql(db, bad)
    with pytest.raises(sqlite3.OperationalError):
        db.execute("INSERT INTO contracts(filename) VALUES ('x')")
