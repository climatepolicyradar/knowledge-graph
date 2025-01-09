def test_analyse_classifier(
    run_just_command, concept_wikibase_id, tmp_path, monkeypatch
):
    """Test the analyse-classifier command"""
    monkeypatch.setattr("scripts.config.data_dir", tmp_path)

    result = run_just_command(f"analyse-classifier {concept_wikibase_id}")

    assert result.returncode == 0
