"""Generate a synthetic contract corpus with ground truth and a golden question set.

Everything is fictional. The corpus is built to contain the traps that break
naive RAG in real contract archives:

* heavy shared boilerplate - clauses say "the Supplier", never the company name,
  so a retrieved chunk alone does not tell you which contract it belongs to;
* near-identical contract numbers (AP-SA-2024-017 vs AP-SA-2024-071) and
  near-identical company names (Harbor Metals vs Harbour Metal Works);
* the same fact written in different formats (dates, amounts, day counts);
* totals that only appear in a fee table;
* one scanned contract without a text layer;
* questions that need *all* contracts (aggregation) or have no answer at all.

Usage:  uv run python scripts/make_corpus.py [--out data/corpus] [--seed 7]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from docx import Document as DocxDocument
from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUR_COMPANY = "Alder Precision GmbH"
OUR_ADDRESS = "Talstrasse 41, 70188 Stuttgart, Germany"

TYPES = {
    "SA": ("Supply Agreement", "Supplier", "the supply of machined components and raw materials"),
    "MSA": ("Master Services Agreement", "Service Provider", "engineering and IT services"),
    "LIC": ("Software License Agreement", "Licensor", "the license of software and related support"),
    "LSA": ("Logistics Services Agreement", "Logistics Provider", "freight, warehousing and customs services"),
    "NDA": ("Mutual Non-Disclosure Agreement", "Recipient", "the exchange of confidential information"),
}

# (name, country, governing law used by default for that country)
COUNTERPARTIES = [
    ("Kestrel Logistics B.V.", "Netherlands"),
    ("Brightwater Analytics Ltd", "United Kingdom"),
    ("Fenwick Polymers Inc.", "United States"),
    ("Harbor Metals Inc.", "United States"),
    ("Harbour Metal Works Ltd", "United Kingdom"),
    ("Lindqvist Automation AB", "Sweden"),
    ("Orrin Software GmbH", "Germany"),
    ("Pellucid Optics SA", "Switzerland"),
    ("Quarry Ridge Castings LLC", "United States"),
    ("Rhenus Valley Fasteners GmbH", "Germany"),
    ("Saltmarsh Cloud Ltd", "United Kingdom"),
    ("Tavira Components Lda", "Portugal"),
    ("Umberfield Electronics Inc.", "United States"),
    ("Vexmoor Freight GmbH", "Germany"),
    ("Wexford Data Systems Ltd", "Ireland"),
    ("Yellowpine Tooling Inc.", "United States"),
    ("Zandvoort Plastics B.V.", "Netherlands"),
    ("Arbelo Engineering S.L.", "Spain"),
    ("Corvane Security GmbH", "Germany"),
    ("Dunmore Packaging Ltd", "United Kingdom"),
    ("Eskerfield Robotics AG", "Switzerland"),
    ("Gallowmere Steel Inc.", "United States"),
    ("Holloway Sensor Labs Inc.", "United States"),
    ("Islay Precision Ltd", "United Kingdom"),
    ("Juniper Coatings GmbH", "Germany"),
    ("Kilnworth Ceramics Ltd", "United Kingdom"),
    ("Larchmont Payroll Services LLC", "United States"),
    ("Marrowby Translation Services Ltd", "Ireland"),
    ("Northgate Test Equipment GmbH", "Germany"),
    ("Ostrava Wire & Cable s.r.o.", "Czech Republic"),
]

LAW_BY_COUNTRY = {
    "Germany": "Germany",
    "United Kingdom": "England and Wales",
    "Ireland": "Ireland",
    "United States": "the State of New York",
    "Netherlands": "the Netherlands",
    "Switzerland": "Switzerland",
}

CURRENCY_BY_COUNTRY = {"United States": "USD", "United Kingdom": "GBP"}

NUMBER_WORDS = {
    10: "ten",
    14: "fourteen",
    15: "fifteen",
    20: "twenty",
    30: "thirty",
    45: "forty-five",
    60: "sixty",
    75: "seventy-five",
    90: "ninety",
    120: "one hundred twenty",
    180: "one hundred eighty",
}

SIGNATORIES = [
    "Anna Keller",
    "Jonas Brandt",
    "Priya Raman",
    "Tom Ellison",
    "Sofia Marques",
    "Lukas Novak",
    "Hannah Whitfield",
    "Daniel Okafor",
    "Marta Lindgren",
    "Oliver Grant",
    "Chloe Dubois",
    "Felix Hartmann",
]


@dataclass
class Contract:
    contract_id: str
    filename: str
    contract_type: str
    counterparty: str
    counterparty_country: str
    effective_date: str
    end_date: str
    auto_renewal: bool
    renewal_notice_days: int | None
    contract_value: float | None
    currency: str | None
    payment_terms_days: int | None
    termination_notice_days: int
    liability_cap: float | None
    governing_law: str
    counterparty_signatory: str
    fmt: str = "pdf"
    scanned: bool = False


# ---------------------------------------------------------------- formatting


def fmt_date(d: dt.date, style: int) -> str:
    return [
        d.strftime("%d %B %Y").lstrip("0"),
        d.strftime("%B %d, %Y").replace(" 0", " "),
        d.isoformat(),
        d.strftime("%d.%m.%Y"),
    ][style]


def fmt_days(n: int, style: int) -> str:
    if style == 0 and n in NUMBER_WORDS:
        return f"{NUMBER_WORDS[n]} ({n}) days"
    return f"{n} days"


def fmt_money(amount: float, currency: str, style: int) -> str:
    if style == 1 and amount >= 1_000_000 and amount % 50_000 == 0:
        millions = f"{amount / 1_000_000:g}"
        word = {"EUR": "euros", "USD": "US dollars", "GBP": "pounds sterling"}[currency]
        return f"{millions} million {word} ({currency} {amount:,.0f})"
    return f"{currency} {amount:,.2f}"


def term_end(start: dt.date, months: int) -> dt.date:
    """Last day of a term of `months` starting on the 1st of a month."""
    y, m = divmod(start.month - 1 + months, 12)
    return dt.date(start.year + y, m + 1, 1) - dt.timedelta(days=1)


# ---------------------------------------------------------------- contracts


def build_contracts(rng: random.Random) -> list[Contract]:
    type_plan = ["SA"] * 9 + ["MSA"] * 6 + ["LIC"] * 5 + ["LSA"] * 5 + ["NDA"] * 5
    rng.shuffle(type_plan)
    used_numbers: set[tuple[str, int, int]] = set()
    contracts = []
    for i, ((name, country), ctype) in enumerate(zip(COUNTERPARTIES, type_plan)):
        start = dt.date(rng.choice([2022, 2023, 2024, 2025]), rng.randint(1, 12), 1)
        term = rng.choice([12, 24, 36]) if ctype != "NDA" else rng.choice([24, 36])
        end = term_end(start, term)
        while True:
            number = rng.randint(1, 120)
            if (ctype, start.year, number) not in used_numbers:
                break
        used_numbers.add((ctype, start.year, number))
        auto = ctype != "NDA" and rng.random() < 0.5
        currency = CURRENCY_BY_COUNTRY.get(country, "EUR")
        value = None if ctype == "NDA" else rng.choice([4, 6, 9, 12, 18, 25, 40, 75, 120, 160]) * 10_000 * term / 12
        cap = None if ctype == "NDA" else rng.choice([250_000, 500_000, 1_000_000, 2_000_000, 2_500_000])
        contracts.append(
            Contract(
                contract_id=f"AP-{ctype}-{start.year}-{number:03d}",
                filename="",
                contract_type=TYPES[ctype][0],
                counterparty=name,
                counterparty_country=country,
                effective_date=start.isoformat(),
                end_date=end.isoformat(),
                auto_renewal=auto,
                renewal_notice_days=rng.choice([30, 60, 90]) if auto else None,
                contract_value=value,
                currency=None if ctype == "NDA" else currency,
                payment_terms_days=None if ctype == "NDA" else rng.choice([30, 45, 60, 90]),
                termination_notice_days=rng.choice([30, 60, 90, 180]),
                liability_cap=cap,
                governing_law=LAW_BY_COUNTRY.get(country, "Germany"),
                counterparty_signatory=SIGNATORIES[i % len(SIGNATORIES)],
            )
        )
    # Trap: two supply agreements with mirrored numbers in the same year.
    sas = [c for c in contracts if c.contract_id.startswith("AP-SA-")]
    sas[0].effective_date = sas[1].effective_date = "2024-03-01"
    sas[0].end_date, sas[1].end_date = "2026-02-28", "2027-02-28"
    sas[0].contract_id, sas[1].contract_id = "AP-SA-2024-017", "AP-SA-2024-071"
    # Two DOCX files and one scanned PDF to show multi-format ingestion.
    contracts[5].fmt = contracts[11].fmt = "docx"
    contracts[-1].scanned = True
    for c in contracts:
        c.filename = f"{c.contract_id}_{c.counterparty.split()[0].replace(',', '')}.{c.fmt}"
    return contracts


def contract_sections(c: Contract, rng: random.Random) -> tuple[str, list[tuple[str, str]], list[list[str]] | None]:
    """Return (preamble, [(heading, body)], fee table rows or None)."""
    code = c.contract_id.split("-")[1]
    title, role, scope = TYPES[code]
    ds = rng.randrange(4)
    eff = dt.date.fromisoformat(c.effective_date)
    end = dt.date.fromisoformat(c.end_date)
    signed = eff - dt.timedelta(days=rng.randint(5, 40))
    party = "Disclosing Party" if code == "NDA" else "Customer"

    preamble = (
        f'This {title} (the "Agreement") is entered into on {fmt_date(signed, ds)} by and between '
        f'{OUR_COMPANY}, {OUR_ADDRESS} (the "{party}"), and {c.counterparty}, a company organized under the '
        f'laws of {c.counterparty_country} (the "{role}"). Each a "Party" and together the "Parties".'
    )
    s: list[tuple[str, str]] = []
    s.append(
        (
            "1. Definitions",
            'In this Agreement, unless the context otherwise requires: "Affiliate" means any entity that directly or '
            'indirectly controls, is controlled by, or is under common control with a Party; "Business Day" means a day '
            'other than a Saturday, Sunday or public holiday in Baden-Wuerttemberg, Germany; "Confidential Information" '
            "means all information disclosed by a Party that is marked as confidential or would reasonably be understood "
            'to be confidential; "Intellectual Property Rights" means patents, utility models, rights in designs, '
            "copyright, database rights, trade marks and all similar rights anywhere in the world. Headings are for "
            "convenience only and do not affect interpretation.",
        )
    )
    s.append(
        (
            "2. Scope",
            f"This Agreement governs {scope} between the Parties. The {role} shall perform its obligations with "
            "reasonable skill, care and diligence, in accordance with good industry practice and all applicable laws. "
            f"Individual orders, statements of work or schedules agreed in writing form part of this Agreement. In case of "
            "conflict, the main body of this Agreement prevails over any schedule, and any schedule prevails over an "
            f"individual order. The {role} shall appoint a contact person responsible for the relationship.",
        )
    )

    term = (
        f'This Agreement shall commence on {fmt_date(eff, ds)} (the "Effective Date") and shall remain in force until {fmt_date(end, ds)}'
    )
    if c.auto_renewal:
        term += (
            ". Thereafter it shall renew automatically for successive periods of twelve (12) months unless either Party "
            f"gives written notice of non-renewal at least {fmt_days(c.renewal_notice_days, rng.randrange(2))} before the "
            "end of the then-current term."
        )
    else:
        term += ", unless terminated earlier in accordance with its terms. This Agreement shall not renew automatically; any extension requires a written amendment signed by both Parties."
    s.append(("3. Term and Renewal", term))

    if code != "NDA":
        pay = rng.randrange(3)
        payment = {
            0: f"The {party} shall pay each undisputed invoice within {fmt_days(c.payment_terms_days, 0)} of receipt.",
            1: f"Payment terms are net {c.payment_terms_days} days from the date of a correct invoice.",
            2: f"Invoices are payable {fmt_days(c.payment_terms_days, 1)} after receipt of a correct invoice.",
        }[pay]
        s.append(
            (
                "4. Fees and Payment",
                f"The fees payable by the {party} are set out in Schedule 1 (Fees). All amounts are stated in {c.currency} and are "
                f"exclusive of value added tax. {payment} Late payments bear interest at the statutory rate. The {role} may "
                "not suspend performance for disputed amounts while the dispute is being resolved in good faith.",
            )
        )
    s.append(
        (
            "5. Confidentiality",
            "Each Party shall keep the Confidential Information of the other Party strictly confidential, use it only for "
            "the purposes of this Agreement and disclose it only to employees, Affiliates and advisers who need to know it "
            "and are bound by equivalent obligations. These obligations do not apply to information that is or becomes "
            "public through no fault of the receiving Party, was lawfully known before disclosure, or must be disclosed by "
            "law. Upon request, the receiving Party shall return or destroy all Confidential Information.",
        )
    )
    if code != "NDA":
        s.append(
            (
                "6. Limitation of Liability",
                "Neither Party excludes or limits liability for death or personal injury caused by negligence, for fraud, or "
                "for any liability that cannot be limited by law. Subject to the foregoing, neither Party shall be liable for "
                "indirect or consequential loss, loss of profit or loss of data. Each Party's total aggregate liability "
                f"arising out of or in connection with this Agreement shall not exceed {fmt_money(c.liability_cap, c.currency, rng.randrange(2))}.",
            )
        )
    s.append(
        (
            "7. Termination",
            f"Either Party may terminate this Agreement for convenience upon {fmt_days(c.termination_notice_days, rng.randrange(2))}' "
            "prior written notice to the other Party. Either Party may terminate this Agreement with immediate effect by written "
            "notice if the other Party commits a material breach which is not remedied within thirty (30) days of being "
            "notified, or becomes insolvent. Termination does not affect accrued rights and remedies.",
        )
    )
    s.append(
        (
            "8. Governing Law and Jurisdiction",
            f"This Agreement and any dispute arising out of it shall be governed by the laws of {c.governing_law}, excluding "
            "its conflict of laws rules and the United Nations Convention on Contracts for the International Sale of Goods. "
            "The courts competent for the registered office of the defendant shall have exclusive jurisdiction.",
        )
    )
    s.append(
        (
            "9. Miscellaneous",
            "Amendments to this Agreement must be made in writing. Neither Party may assign this Agreement without the prior "
            "written consent of the other Party, which shall not be unreasonably withheld. If any provision is held invalid, "
            "the remaining provisions remain in full force. This Agreement constitutes the entire agreement between the "
            "Parties regarding its subject matter and supersedes all prior understandings.",
        )
    )
    s.append(
        (
            "Signatures",
            f"For {OUR_COMPANY}: Markus Ebner, Head of Procurement.    For {c.counterparty}: {c.counterparty_signatory}, Authorized Signatory.",
        )
    )

    table = None
    if c.contract_value is not None:
        years = max(1, round((end - eff).days / 365))
        per_year = c.contract_value / years
        table = [["Year", "Period", f"Amount ({c.currency})"]]
        for y in range(years):
            p_start = eff.replace(year=eff.year + y)
            p_end = min(end, p_start.replace(year=p_start.year + 1) - dt.timedelta(days=1))
            table.append([str(y + 1), f"{p_start.isoformat()} to {p_end.isoformat()}", f"{per_year:,.2f}"])
        table.append(["Total", "", f"{c.contract_value:,.2f}"])
    return preamble, s, table


# ---------------------------------------------------------------- writers


class ContractPDF(FPDF):
    def __init__(self, footer_text: str):
        super().__init__(format="A4")
        self.footer_text = footer_text

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 5, f"{self.footer_text} - page {self.page_no()}", align="C")


def write_pdf(path: Path, c: Contract, preamble: str, sections, table) -> None:
    pdf = ContractPDF(f"{c.contract_id} - confidential")
    pdf.set_auto_page_break(True, margin=18)
    pdf.set_margins(22, 20, 22)
    pdf.add_page()
    title = TYPES[c.contract_id.split("-")[1]][0].upper()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Contract No.: {c.contract_id}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)
    pdf.multi_cell(0, 5.2, preamble)
    for heading, body in sections:
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, heading, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5.2, body)
    if table:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Schedule 1 - Fees", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5.2, "The fees for the term of this Agreement are as follows:")
        pdf.ln(2)
        with pdf.table(col_widths=(20, 80, 50), text_align=("LEFT", "LEFT", "RIGHT")) as t:
            for row in table:
                r = t.row()
                for cell in row:
                    r.cell(cell)
    pdf.output(str(path))


def write_docx(path: Path, c: Contract, preamble: str, sections, table) -> None:
    doc = DocxDocument()
    doc.add_heading(TYPES[c.contract_id.split("-")[1]][0].upper(), level=0)
    doc.add_paragraph(f"Contract No.: {c.contract_id}")
    doc.add_paragraph(preamble)
    for heading, body in sections:
        doc.add_heading(heading, level=2)
        doc.add_paragraph(body)
    if table:
        doc.add_heading("Schedule 1 - Fees", level=2)
        t = doc.add_table(rows=0, cols=3)
        for row in table:
            cells = t.add_row().cells
            for cell, value in zip(cells, row):
                cell.text = value
    doc.save(str(path))


def wrap(draw: ImageDraw.ImageDraw, text: str, font, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) > width:
            lines.append(line)
            line = word
        else:
            line = trial
    return lines + [line]


def write_scanned_pdf(path: Path, c: Contract, preamble: str, sections, table, rng: random.Random) -> None:
    """Render the contract as page images only - no text layer, like a scan."""
    W, H, margin = 1240, 1754, 110  # A4 at 150 dpi
    body = ImageFont.load_default(size=22)
    bold = ImageFont.load_default(size=26)
    blocks = [(TYPES[c.contract_id.split("-")[1]][0].upper(), bold), (f"Contract No.: {c.contract_id}", body), (preamble, body)]
    for heading, text in sections:
        blocks += [(heading, bold), (text, body)]
    if table:
        blocks.append(("Schedule 1 - Fees", bold))
        blocks += [("    ".join(row), body) for row in table]
    pages, img, y = [], None, H
    for text, font in blocks:
        probe = ImageDraw.Draw(Image.new("L", (1, 1)))
        for line in wrap(probe, text, font, W - 2 * margin):
            if y > H - margin:
                img = Image.new("L", (W, H), 250)
                pages.append(img)
                y = margin
            ImageDraw.Draw(img).text((margin, y), line, fill=25, font=font)
            y += 32
        y += 14
    pdf = FPDF(format="A4")
    for img in pages:
        noisy = img.rotate(rng.uniform(-0.6, 0.6), resample=Image.BICUBIC, fillcolor=250).filter(ImageFilter.GaussianBlur(0.6))
        pdf.add_page()
        pdf.image(noisy, x=0, y=0, w=210, h=297)  # PIL image directly: fpdf caches images by file path
    pdf.output(str(path))


# ---------------------------------------------------------------- golden set


def law_aliases(law: str) -> list[str]:
    return {
        "Germany": ["German", "Germany"],
        "England and Wales": ["England"],
        "Ireland": ["Ireland", "Irish"],
        "the State of New York": ["New York"],
        "the Netherlands": ["Netherlands", "Dutch"],
        "Switzerland": ["Switzerland", "Swiss"],
    }[law]


def build_golden(contracts: list[Contract], rng: random.Random) -> list[dict]:
    qs: list[dict] = []
    readable = [c for c in contracts if not c.scanned]
    by_id = {c.contract_id: c for c in contracts}

    def lookup(c: Contract, question: str, kind: str, value, tag: str):
        qs.append(
            {
                "id": f"L{len(qs) + 1:02d}",
                "type": "lookup",
                "tag": tag,
                "question": question,
                "expected": {"kind": kind, "value": value},
                "source": c.filename,
            }
        )

    # Keyword trap: mirrored contract numbers.
    for cid in ("AP-SA-2024-017", "AP-SA-2024-071"):
        lookup(by_id[cid], f"When does contract {cid} end?", "date", by_id[cid].end_date, "contract-id")

    picks = rng.sample(readable, 22)
    templates = [
        ("end_date", "date", "When does the {type} with {cp} expire?", "name"),
        ("payment_terms_days", "number", "What are the payment terms in contract {id}?", "contract-id"),
        ("governing_law", "text", "Which law governs the agreement with {cp}?", "name"),
        ("termination_notice_days", "number", "How many days' notice does {cp} need to give to terminate for convenience?", "name"),
        ("liability_cap", "number", "What is the liability cap in the agreement with {cp}?", "name"),
        ("contract_value", "number", "What is the total contract value of {id}?", "contract-id"),
        (
            "renewal_notice_days",
            "number",
            "How many days before the end of the term must a non-renewal notice be given for contract {id}?",
            "contract-id",
        ),
        ("effective_date", "date", "What is the effective date of the agreement with {cp}?", "name"),
        ("counterparty_signatory", "text", "Who signed contract {id} on behalf of the counterparty?", "contract-id"),
    ]
    ti = 0
    for c in picks:
        for _ in range(len(templates)):
            field, kind, tpl, tag = templates[ti % len(templates)]
            ti += 1
            value = getattr(c, field)
            if value is not None:
                break
        else:
            continue
        if field == "governing_law":
            value = law_aliases(value)
        q = tpl.format(type=c.contract_type.lower().replace("mutual ", ""), cp=c.counterparty, id=c.contract_id)
        lookup(c, q, kind, value, tag)
    # Near-identical names.
    for name in ("Harbor Metals Inc.", "Harbour Metal Works Ltd"):
        c = next(c for c in contracts if c.counterparty == name)
        lookup(c, f"What is the governing law of the contract with {name}?", "text", law_aliases(c.governing_law), "similar-name")

    def agg(question: str, ids=None, value=None):
        exp = {"kind": "ids", "value": sorted(ids)} if ids is not None else {"kind": "number", "value": value}
        qs.append(
            {
                "id": f"A{sum(q['type'] == 'aggregate' for q in qs) + 1:02d}",
                "type": "aggregate",
                "tag": "aggregate",
                "question": question,
                "expected": exp,
            }
        )

    agg("List all contracts that end in 2027.", [c.contract_id for c in contracts if c.end_date.startswith("2027")])
    agg("Which contracts are governed by German law?", [c.contract_id for c in contracts if c.governing_law == "Germany"])
    agg("Which contracts renew automatically?", [c.contract_id for c in contracts if c.auto_renewal])
    agg("How many supply agreements do we have?", value=sum(c.contract_type == "Supply Agreement" for c in contracts))
    agg(
        "List all contracts with payment terms longer than 45 days.", [c.contract_id for c in contracts if (c.payment_terms_days or 0) > 45]
    )
    agg(
        "Which contracts with counterparties in the United States end before 1 January 2027?",
        [c.contract_id for c in contracts if c.counterparty_country == "United States" and c.end_date < "2027-01-01"],
    )
    agg(
        "Which contracts require at least 90 days' notice of non-renewal?",
        [c.contract_id for c in contracts if (c.renewal_notice_days or 0) >= 90],
    )
    agg(
        "What is the total value of all EUR-denominated contracts?",
        value=sum(c.contract_value for c in contracts if c.currency == "EUR" and c.contract_value),
    )
    agg(
        "List all contracts with a liability cap of at least 2,000,000.",
        [c.contract_id for c in contracts if (c.liability_cap or 0) >= 2_000_000],
    )
    agg(
        "How many contracts can be terminated for convenience with 30 days' notice or less?",
        value=sum(c.termination_notice_days <= 30 for c in contracts),
    )

    nda = next(c for c in readable if c.contract_type.endswith("Non-Disclosure Agreement"))
    sa = by_id["AP-SA-2024-017"]
    unanswerable = [
        "What are the payment terms in our agreement with Globex Corporation?",
        f"What is the liability cap in the non-disclosure agreement with {nda.counterparty}?",
        f"What penalty applies for late delivery under contract {sa.contract_id}?",
        "What minimum insurance coverage must Kestrel Logistics B.V. maintain?",
        f"What is the unit price per component in contract {sa.contract_id}?",
        "Which contract covers the office cleaning services in Munich?",
        f"What is the data retention period after termination for {picks[0].counterparty}?",
        "When was the agreement with Northwind Traders signed?",
    ]
    for i, q in enumerate(unanswerable, 1):
        qs.append({"id": f"U{i:02d}", "type": "unanswerable", "tag": "unanswerable", "question": q, "expected": {"kind": "refusal"}})
    return qs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/corpus")
    ap.add_argument("--golden", default="evals/golden.yaml")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("AP-*"):
        old.unlink()

    contracts = build_contracts(rng)
    for c in contracts:
        preamble, sections, table = contract_sections(c, rng)
        path = out / c.filename
        if c.fmt == "docx":
            write_docx(path, c, preamble, sections, table)
        elif c.scanned:
            write_scanned_pdf(path, c, preamble, sections, table, rng)
        else:
            write_pdf(path, c, preamble, sections, table)

    truth = [asdict(c) for c in contracts]
    (out / "ground_truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    golden = build_golden(contracts, rng)
    Path(args.golden).parent.mkdir(parents=True, exist_ok=True)
    header = "# Generated by scripts/make_corpus.py - do not edit by hand.\n"
    Path(args.golden).write_text(header + yaml.safe_dump(golden, sort_keys=False, allow_unicode=True, width=120), encoding="utf-8")
    counts = {t: sum(q["type"] == t for q in golden) for t in ("lookup", "aggregate", "unanswerable")}
    print(f"{len(contracts)} contracts -> {out}; golden set {counts} -> {args.golden}")


if __name__ == "__main__":
    main()
