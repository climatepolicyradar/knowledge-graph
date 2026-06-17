from unittest.mock import Mock, patch

import pytest
import typer

from knowledge_graph.cloud import AwsEnv, ClassifierSpec
from knowledge_graph.identifiers import WikibaseID
from scripts.deploy import existing, new


def _make_mock_classifier(classifier_id: str = "clf-abc") -> Mock:
    clf = Mock()
    clf.id = classifier_id
    return clf


@patch("scripts.deploy.refresh_all_available_classifiers")
@patch("scripts.deploy.scripts.promote.main")
@patch("scripts.deploy.scripts.train.main")
@patch("scripts.deploy.parse_spec_file")
@patch("scripts.deploy.validate_transition")
def test_existing_trains_and_promotes_each_spec(
    mock_validate, mock_parse_spec, mock_train, mock_promote, mock_refresh
):
    specs = [
        ClassifierSpec(name="Q100", alias="v1"),
        ClassifierSpec(name="Q200", alias="v2"),
    ]
    mock_parse_spec.return_value = specs
    mock_train.return_value = _make_mock_classifier("clf-1")

    existing(
        from_aws_env=AwsEnv.staging,
        to_aws_env=AwsEnv.production,
        train=True,
        promote=True,
    )

    assert mock_train.call_count == 2
    assert mock_promote.call_count == 2


@patch("scripts.deploy.refresh_all_available_classifiers")
@patch("scripts.deploy.scripts.promote.main")
@patch("scripts.deploy.scripts.train.main")
@patch("scripts.deploy.parse_spec_file")
@patch("scripts.deploy.validate_transition")
def test_existing_skips_promote_when_promote_is_false(
    mock_validate, mock_parse_spec, mock_train, mock_promote, mock_refresh
):
    mock_parse_spec.return_value = [ClassifierSpec(name="Q100", alias="v1")]
    mock_train.return_value = _make_mock_classifier()

    existing(
        from_aws_env=AwsEnv.staging,
        to_aws_env=AwsEnv.production,
        train=True,
        promote=False,
    )

    mock_train.assert_called_once()
    mock_promote.assert_not_called()


@patch("scripts.deploy.refresh_all_available_classifiers")
@patch("scripts.deploy.scripts.promote.main")
@patch("scripts.deploy.scripts.train.main")
@patch("scripts.deploy.parse_spec_file")
@patch("scripts.deploy.validate_transition")
def test_existing_skips_train_and_promote_when_both_false(
    mock_validate, mock_parse_spec, mock_train, mock_promote, mock_refresh
):
    mock_parse_spec.return_value = [ClassifierSpec(name="Q100", alias="v1")]

    existing(
        from_aws_env=AwsEnv.staging,
        to_aws_env=AwsEnv.production,
        train=False,
        promote=False,
    )

    mock_train.assert_not_called()
    mock_promote.assert_not_called()


@patch("scripts.deploy.refresh_all_available_classifiers")
@patch("scripts.deploy.scripts.promote.main")
@patch("scripts.deploy.scripts.train.main")
@patch("scripts.deploy.parse_spec_file")
@patch("scripts.deploy.validate_transition")
def test_existing_calls_refresh_after_processing(
    mock_validate, mock_parse_spec, mock_train, mock_promote, mock_refresh
):
    mock_parse_spec.return_value = [ClassifierSpec(name="Q100", alias="v1")]
    mock_train.return_value = _make_mock_classifier()

    existing(
        from_aws_env=AwsEnv.staging,
        to_aws_env=AwsEnv.production,
        train=True,
        promote=True,
    )

    mock_refresh.assert_called_once_with([AwsEnv.production])


@patch("scripts.deploy.refresh_all_available_classifiers")
@patch("scripts.deploy.scripts.promote.main")
@patch("scripts.deploy.scripts.train.main")
@patch("scripts.deploy.parse_spec_file")
@patch("scripts.deploy.validate_transition")
def test_existing_raises_when_train_returns_no_classifier(
    mock_validate, mock_parse_spec, mock_train, mock_promote, mock_refresh
):
    mock_parse_spec.return_value = [ClassifierSpec(name="Q100", alias="v1")]
    mock_train.return_value = None

    with pytest.raises(ValueError, match="No classifier returned"):
        existing(
            from_aws_env=AwsEnv.staging,
            to_aws_env=AwsEnv.production,
            train=True,
            promote=True,
        )


@patch("scripts.deploy.refresh_all_available_classifiers")
@patch("scripts.deploy.scripts.promote.main")
@patch("scripts.deploy.scripts.train.main")
@patch("scripts.deploy.parse_spec_file")
@patch("scripts.deploy.validate_transition")
def test_existing_raises_for_duplicate_add_remove_profiles(
    mock_validate, mock_parse_spec, mock_train, mock_promote, mock_refresh
):
    mock_parse_spec.return_value = []

    with pytest.raises(typer.BadParameter, match="duplicate"):
        existing(
            from_aws_env=AwsEnv.staging,
            to_aws_env=AwsEnv.production,
            train=False,
            promote=False,
            add_classifiers_profiles=["profile-a"],
            remove_classifiers_profiles=["profile-a"],
        )


@patch("scripts.deploy.refresh_all_available_classifiers")
@patch("scripts.deploy.scripts.promote.main")
@patch("scripts.deploy.scripts.train.main")
def test_new_trains_and_promotes_each_wikibase_id(
    mock_train, mock_promote, mock_refresh
):
    mock_train.return_value = _make_mock_classifier()

    new(
        aws_env=AwsEnv.staging,
        wikibase_ids=[WikibaseID("Q100"), WikibaseID("Q200")],
        train=True,
        promote=True,
    )

    assert mock_train.call_count == 2
    assert mock_promote.call_count == 2


@patch("scripts.deploy.refresh_all_available_classifiers")
@patch("scripts.deploy.scripts.promote.main")
@patch("scripts.deploy.scripts.train.main")
def test_new_skips_promote_when_promote_is_false(
    mock_train, mock_promote, mock_refresh
):
    mock_train.return_value = _make_mock_classifier()

    new(
        aws_env=AwsEnv.staging,
        wikibase_ids=[WikibaseID("Q100")],
        train=True,
        promote=False,
    )

    mock_train.assert_called_once()
    mock_promote.assert_not_called()


@patch("scripts.deploy.refresh_all_available_classifiers")
@patch("scripts.deploy.scripts.promote.main")
@patch("scripts.deploy.scripts.train.main")
def test_new_calls_refresh_after_processing(mock_train, mock_promote, mock_refresh):
    mock_train.return_value = _make_mock_classifier()

    new(
        aws_env=AwsEnv.staging,
        wikibase_ids=[WikibaseID("Q100")],
        train=True,
        promote=True,
    )

    mock_refresh.assert_called_once_with([AwsEnv.staging])


@patch("scripts.deploy.refresh_all_available_classifiers")
@patch("scripts.deploy.scripts.promote.main")
@patch("scripts.deploy.scripts.train.main")
def test_new_exits_with_code_1_when_train_raises_attribute_error(
    mock_train, mock_promote, mock_refresh
):
    mock_train.side_effect = AttributeError("boom")

    with pytest.raises(typer.Exit) as exc_info:
        new(
            aws_env=AwsEnv.staging,
            wikibase_ids=[WikibaseID("Q100")],
            train=True,
            promote=True,
        )

    assert exc_info.value.exit_code == 1


@patch("scripts.deploy.refresh_all_available_classifiers")
@patch("scripts.deploy.scripts.promote.main")
@patch("scripts.deploy.scripts.train.main")
def test_new_continues_processing_remaining_ids_after_attribute_error(
    mock_train, mock_promote, mock_refresh
):
    mock_train.side_effect = [AttributeError("boom"), _make_mock_classifier("clf-2")]

    with pytest.raises(typer.Exit):
        new(
            aws_env=AwsEnv.staging,
            wikibase_ids=[WikibaseID("Q100"), WikibaseID("Q200")],
            train=True,
            promote=True,
        )

    assert mock_train.call_count == 2
    mock_promote.assert_called_once()


@patch("scripts.deploy.refresh_all_available_classifiers")
@patch("scripts.deploy.scripts.promote.main")
@patch("scripts.deploy.scripts.train.main")
def test_new_raises_bad_parameter_for_multiple_add_classifiers_profiles(
    mock_train, mock_promote, mock_refresh
):
    with pytest.raises(typer.BadParameter):
        new(
            aws_env=AwsEnv.staging,
            wikibase_ids=[WikibaseID("Q100")],
            train=True,
            promote=True,
            add_classifiers_profiles=["profile-a", "profile-b"],
        )
