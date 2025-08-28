import pytest


@pytest.mark.skip(
    reason="Relies on the full feather file which is not pushed to the repo"
)
def test_analyse_classifier(
    run_just_command, concept_wikibase_id, tmp_path, monkeypatch
):
    """Test the analyse-classifier command"""
    monkeypatch.setattr("src.config.data_dir", tmp_path)

    result = run_just_command(f"analyse-classifier {concept_wikibase_id}")

    assert result.returncode == 0
