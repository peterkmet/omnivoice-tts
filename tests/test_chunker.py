import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chunker import chunk_text


INPUT_TEXT = """\
SERGEJ LUKJANĚNKO - ŠERÁ HLÍDKA

Tento text je pro věc Světla irelevantní.

Noční hlídka

Tento text je pro věc Tmy irelevantní.

Denní hlídka

Část první

Ničí čas

Prolog

Opravdické moskevské dvory vymizely kdysi v dobách mezi Vysockým a Okudžavou.

Je to zvláštní. Dokonce i po revoluci, kdy se v zájmu odstranění kuchyňské otročiny žen po domech likvidovaly kuchyně, na dvory naštěstí nikdo neútočil.
"""

EXPECTED_CHUNKS = [
    (
        "SERGEJ LUKJANĚNKO - ŠERÁ HLÍDKA. "
        "Tento text je pro věc Světla irelevantní. "
        "Noční hlídka. "
        "Tento text je pro věc Tmy irelevantní. "
        "Denní hlídka. "
        "Část první. "
        "Ničí čas. "
        "Prolog. "
        "Opravdické moskevské dvory vymizely kdysi v dobách mezi Vysockým a Okudžavou. "
        "Je to zvláštní. Dokonce i po revoluci, kdy se v zájmu odstranění kuchyňské otročiny žen po domech likvidovaly kuchyně, na dvory naštěstí nikdo neútočil."
    )
]


class TestGoldenOutput:
    def test_real_book_text(self):
        chunks = chunk_text(INPUT_TEXT, chunk_size=10240)
        assert chunks == EXPECTED_CHUNKS


class TestBasicChunking:
    def test_empty_text(self):
        assert chunk_text("") == []

    def test_whitespace_only(self):
        assert chunk_text("   \n\n  ") == []

    def test_single_paragraph_fits(self):
        text = "Hello world. This is a test."
        chunks = chunk_text(text, chunk_size=1024)
        assert len(chunks) == 1
        assert chunks[0] == text


class TestParagraphBoundaries:
    def test_splits_at_paragraph_boundary(self):
        p1 = "First paragraph with some text."
        p2 = "Second paragraph with more text."
        text = p1 + "\n\n" + p2
        chunks = chunk_text(text, chunk_size=len(p1.encode()) + 5)
        assert len(chunks) == 2
        assert chunks[0] == p1
        assert chunks[1] == p2

    def test_never_splits_mid_paragraph(self):
        long_para = ("word " * 200).strip()
        chunks = chunk_text(long_para, chunk_size=100)
        assert len(chunks) == 1

    def test_multiple_paragraphs_packed_into_one_chunk(self):
        paragraphs = ["Short para."] * 10
        text = "\n\n".join(paragraphs)
        chunks = chunk_text(text, chunk_size=10000)
        assert len(chunks) == 1

    def test_paragraphs_joined_with_space(self):
        p1 = "Para one."
        p2 = "Para two."
        text = p1 + "\n\n" + p2
        chunks = chunk_text(text, chunk_size=10000)
        assert len(chunks) == 1
        assert chunks[0] == "Para one. Para two."

    def test_blank_paragraphs_ignored(self):
        text = "Para one.\n\n\n\nPara two."
        chunks = chunk_text(text, chunk_size=10000)
        assert len(chunks) == 1
        assert "Para one." in chunks[0]
        assert "Para two." in chunks[0]

    def test_three_paragraphs_split_one_each(self):
        p1 = "A" * 80
        p2 = "B" * 80
        p3 = "C" * 80
        text = "\n\n".join([p1, p2, p3])
        chunks = chunk_text(text, chunk_size=100)
        assert len(chunks) == 3
        assert chunks[0] == p1 + "."
        assert chunks[1] == p2 + "."
        assert chunks[2] == p3 + "."

    def test_period_added_to_paragraph_without_punctuation(self):
        text = "Noční hlídka\n\nDalší odstavec."
        chunks = chunk_text(text, chunk_size=10000)
        assert chunks[0].startswith("Noční hlídka.")

    def test_period_not_doubled_on_existing_punctuation(self):
        text = "Prvý odstavec.\n\nDruhý odstavec."
        chunks = chunk_text(text, chunk_size=10000)
        assert "Prvý odstavec.." not in chunks[0]


class TestAllTextPreserved:
    def test_all_paragraphs_present(self):
        paragraphs = [f"Paragraph {i} with some content." for i in range(20)]
        text = "\n\n".join(paragraphs)
        chunks = chunk_text(text, chunk_size=80)
        reassembled = " ".join(chunks)
        for p in paragraphs:
            assert p in reassembled

    def test_no_text_lost_or_duplicated(self):
        paragraphs = [f"Text block {i}." for i in range(10)]
        text = "\n\n".join(paragraphs)
        chunks = chunk_text(text, chunk_size=60)
        all_text = " ".join(chunks)
        for i in range(10):
            assert all_text.count(f"Text block {i}.") == 1


class TestUnicode:
    def test_multibyte_characters(self):
        paragraphs = ["Príšerné žlté kone bežali poľom."] * 20
        text = "\n\n".join(paragraphs)
        chunks = chunk_text(text, chunk_size=200)
        assert len(chunks) > 1
        for chunk in chunks:
            chunk.encode("utf-8")

    def test_czech_text(self):
        paragraphs = [f"Červený kůň skákal přes potok číslo {i}." for i in range(10)]
        text = "\n\n".join(paragraphs)
        chunks = chunk_text(text, chunk_size=100)
        assert len(chunks) > 1
        reassembled = " ".join(chunks)
        for i in range(10):
            assert f"přes potok číslo {i}" in reassembled
