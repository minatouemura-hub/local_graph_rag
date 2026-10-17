from __future__ import annotations

import json
from pathlib import Path

from local_graphrag.domain.models import BoundingBox, OCRBlock, OCRDocument, OCRPage, SourceDocument
from local_graphrag.services.layout_chunker import LayoutAwareChunker

FIXTURE = Path(__file__).parents[1] / "fixtures" / "sample_ocr_document.json"


def load_fixture() -> OCRDocument:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    pages = tuple(
        OCRPage(
            document_id=raw["document_id"],
            page=page["page"],
            width=None,
            height=None,
            blocks=tuple(
                OCRBlock(
                    block_id=block["block_id"],
                    page=page["page"],
                    text=block["text"],
                    bbox=BoundingBox(*block["bbox"]) if block.get("bbox") else None,
                    layout_type=block["layout_type"],
                    order=block["order"],
                )
                for block in page["blocks"]
            ),
        )
        for page in raw["pages"]
    )
    return OCRDocument(SourceDocument(raw["document_id"], Path(raw["path"])), pages)


def test_section_headers_create_distinct_sections() -> None:
    chunks = LayoutAwareChunker().chunk(load_fixture())
    section_ids = {chunk.section_id for chunk in chunks if chunk.metadata.get("section_title")}
    assert "manual-001:section:1" in section_ids
    assert "manual-001:section:2" in section_ids
    assert len(section_ids) >= 2


def test_text_blocks_merge_within_section_and_page() -> None:
    chunks = LayoutAwareChunker().chunk(load_fixture())
    merged = next(chunk for chunk in chunks if "本装置を使用する前に" in chunk.text)
    assert "必ず電源を切ってください。" in merged.text
    assert merged.page == 1
    assert merged.metadata["source_block_ids"] == ("b-t1", "b-t2")


def test_table_and_formula_are_independent_chunks() -> None:
    chunks = LayoutAwareChunker().chunk(load_fixture())
    table = next(chunk for chunk in chunks if chunk.layout_types == ("Table",))
    formula = next(chunk for chunk in chunks if chunk.layout_types == ("Formula",))
    assert table.text.startswith("<table>")
    assert formula.text == "P = VI"
    assert table.chunk_id != formula.chunk_id


def test_list_items_merge() -> None:
    chunks = LayoutAwareChunker().chunk(load_fixture())
    list_chunk = next(chunk for chunk in chunks if chunk.layout_types == ("List-item",))
    assert list_chunk.text == "水に濡らさない\n分解しない"
    assert list_chunk.metadata["source_block_ids"] == ("b-l1", "b-l2")


def test_page_boundary_is_preserved_even_for_same_section() -> None:
    document = load_fixture()
    chunks = LayoutAwareChunker().chunk(document)
    page_two_text = next(chunk for chunk in chunks if chunk.text == "次に起動ボタンを押します。")
    assert page_two_text.page == 2
    assert not any(chunk.page == 1 and "次に起動" in chunk.text for chunk in chunks)


def test_bbox_union_and_reading_order() -> None:
    chunks = LayoutAwareChunker().chunk(load_fixture())
    merged = next(chunk for chunk in chunks if "本装置を使用する前に" in chunk.text)
    assert merged.bbox == BoundingBox(10, 120, 250, 175)
    assert [chunk.order for chunk in chunks] == list(range(len(chunks)))


def test_deterministic_ids_order_and_text() -> None:
    document = load_fixture()
    first = LayoutAwareChunker().chunk(document)
    second = LayoutAwareChunker().chunk(document)
    assert [(c.chunk_id, c.order, c.text) for c in first] == [(c.chunk_id, c.order, c.text) for c in second]


def test_empty_text_is_ignored_but_empty_figure_is_retained() -> None:
    document = OCRDocument(
        SourceDocument("empty-001", Path("empty.pdf")),
        (OCRPage("empty-001", 1, None, None, (
            OCRBlock("empty-text", 1, "  ", None, "Text", 0),
            OCRBlock("empty-figure", 1, "", None, "Figure", 1),
        )),),
    )
    chunks = LayoutAwareChunker().chunk(document)
    assert len(chunks) == 1
    assert chunks[0].layout_types == ("Figure",)
    assert chunks[0].metadata["source_block_ids"] == ("empty-figure",)


def test_long_text_is_split_without_text_loss() -> None:
    text = "0123456789" * 20
    document = OCRDocument(
        SourceDocument("long-001", Path("long.pdf")),
        (OCRPage("long-001", 1, None, None, (OCRBlock("long", 1, text, None, "Text", 0),)),),
    )
    chunks = LayoutAwareChunker(max_tokens=10, chars_per_token=1).chunk(document)
    assert len(chunks) == 20
    assert "".join(chunk.text for chunk in chunks) == text


def test_blocks_are_sorted_by_page_then_reading_order() -> None:
    document = OCRDocument(
        SourceDocument("order-001", Path("order.pdf")),
        (OCRPage("order-001", 2, None, None, (OCRBlock("p2", 2, "second", None, "Text", 0),)),
         OCRPage("order-001", 1, None, None, (OCRBlock("p1", 1, "first", None, "Text", 99),))),
    )
    chunks = LayoutAwareChunker().chunk(document)
    assert [chunk.text for chunk in chunks] == ["first", "second"]
