import pytest

from evaluation.labeling.dataset import fingerprint, load_conversations, message_id, parse_id

HEADER = "conversation_key,conversation_message_number,part_number,author,message\n"


def _csv(tmp_path, body: str, bom: bool = True):
    path = tmp_path / "messages.csv"
    path.write_text(("﻿" if bom else "") + HEADER + body, encoding="utf-8")
    return path


def test_load_groups_and_orders(tmp_path):
    csv = _csv(tmp_path, "b,1,1,Bo,hey\na,2,1,Al,two\na,1,1,Al,one\na,3,1,Cy,three\n")
    convs = load_conversations(csv)
    assert list(convs) == ["a", "b"]
    assert [m.number for m in convs["a"]] == [1, 2, 3]
    assert convs["a"][0].id == "a:1"
    assert convs["a"][2].author == "Cy"


def test_load_handles_quotes_bom_and_escapes(tmp_path):
    csv = _csv(tmp_path, 'a,1,1,Al,"hi, ""you""\\nok 😀"\n')
    convs = load_conversations(csv)
    assert convs["a"][0].content == 'hi, "you"\nok 😀'


def test_rejects_part_number_not_1(tmp_path):
    with pytest.raises(ValueError, match="part_number"):
        load_conversations(_csv(tmp_path, "a,1,2,Al,x\n"))


def test_rejects_gap_in_numbers(tmp_path):
    with pytest.raises(ValueError, match="contiguous"):
        load_conversations(_csv(tmp_path, "a,1,1,Al,x\na,3,1,Al,y\n"))


def test_rejects_duplicate(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        load_conversations(_csv(tmp_path, "a,1,1,Al,x\na,1,1,Al,y\n"))


def test_parse_id_roundtrip():
    assert parse_id(message_id("298954459172700181", 12)) == ("298954459172700181", 12)


def test_fingerprint_changes_with_content(tmp_path):
    csv = _csv(tmp_path, "a,1,1,Al,x\n")
    before = fingerprint(csv)
    csv.write_text(csv.read_text(encoding="utf-8").replace(",x", ",y"), encoding="utf-8")
    assert fingerprint(csv) != before
    assert len(before) == 64
