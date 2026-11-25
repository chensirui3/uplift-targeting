"""The demo page in docs/ shows the committed results, and nothing else."""

import json

import pytest

from uplift_targeting import demo


@pytest.mark.parametrize("name", list(demo.EXPORTS))
def test_demo_data_is_what_the_export_builds_from_results(name):
    # Building also checks the page's arithmetic against the results it reads.
    committed = json.loads(demo.output_path(name).read_text(encoding="utf-8"))
    assert committed == json.loads(json.dumps(demo.EXPORTS[name]()))


def test_export_refuses_results_the_page_would_misstate(monkeypatch):
    read = demo._read_json

    def tampered(name, *args, **kwargs):
        payload = read(name, *args, **kwargs)
        if name == "summary.json":
            payload["email_everyone"]["profit_per_1000"]["0.10"]["estimate"] += 5
        return payload

    monkeypatch.setattr(demo, "_read_json", tampered)
    with pytest.raises(ValueError, match="doesn't match summary.json"):
        demo.build_hillstrom()


def test_criteo_export_refuses_a_gain_over_random_its_curve_contradicts(monkeypatch):
    # The page draws random targeting as budget x the effect of treating everyone.
    read = demo._read_json

    def tampered(name, *args, **kwargs):
        payload = read(name, *args, **kwargs)
        if name == "summary.json" and "conversion" in payload:
            payload["conversion"]["reweighted"]["models"]["chosen"]["10pct"]["vs_random"] += 5
        return payload

    monkeypatch.setattr(demo, "_read_json", tampered)
    with pytest.raises(ValueError, match="doesn't match the curve"):
        demo.build_criteo()
