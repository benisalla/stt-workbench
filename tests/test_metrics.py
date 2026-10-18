from stt.metrics import cer, edit_alignment, edit_stats, error_report, format_alignment, main, wer


def test_edit_stats():
    s = edit_stats(["the", "cat", "sat"], ["the", "bat", "sat", "down"])
    assert (s.substitutions, s.deletions, s.insertions, s.ref_length) == (1, 0, 1, 3)
    assert abs(s.rate - 2 / 3) < 1e-9


def test_wer_cer_values():
    assert wer(["a b c"], ["a b c"]) == 0.0
    assert wer(["a b c"], [""]) == 1.0
    assert abs(cer(["abc"], ["abd"]) - 1 / 3) < 1e-9
    assert wer(["a", "b c"], ["x", "b c"]) == 1 / 3  # corpus-level, not averaged


def test_alignment_ops_cover_both_sequences():
    ops = edit_alignment(list("kitten"), list("sitting"))
    assert sum(op != "I" for op, _, _ in ops) == 6
    assert sum(op != "D" for op, _, _ in ops) == 7
    assert sum(op != "=" for op, _, _ in ops) == 3


def test_format_and_report(tmp_path, capsys):
    text = format_alignment("the cat sat", "the bat")
    assert text.startswith("REF:") and "***" in text
    report = error_report(["a b", "c"], ["a b", "d"])
    assert report["wer"] == 1 / 3 and report["per_utterance_wer"] == [0.0, 1.0]
    (tmp_path / "ref.txt").write_text("a b\nc\n")
    (tmp_path / "hyp.txt").write_text("a b\nd\n")
    main([str(tmp_path / "ref.txt"), str(tmp_path / "hyp.txt"), "--worst", "1", "--json", str(tmp_path / "r.json")])
    assert "WER 33.33%" in capsys.readouterr().out
    assert (tmp_path / "r.json").exists()
