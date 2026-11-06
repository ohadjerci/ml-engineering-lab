"""Selection must never replace the alias with a worse or unmeasured model."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from mlflow.exceptions import MlflowException

from src.utils.model_registry import promote_if_better


@pytest.mark.parametrize("candidate,expected", [(0.9, True), (1.0, False), (1.1, False)])
def test_promote_only_strict_improvement(candidate, expected):
    client = Mock()
    client.get_registered_model.return_value = SimpleNamespace(aliases={"best_model": "1"})
    client.get_model_version.return_value = SimpleNamespace(
        tags={"selection_metric": "test/loss", "selection_score": "1.0"}
    )
    assert (
        promote_if_better(client, "MNIST", "best_model", "2", "test/loss", candidate) is expected
    )
    assert client.set_registered_model_alias.called is expected


def test_first_version_is_promoted():
    client = Mock()
    client.get_registered_model.return_value = SimpleNamespace(aliases={})
    assert promote_if_better(client, "MNIST", "best_model", "1", "test/loss", 1.0)
    client.set_registered_model_alias.assert_called_once_with("MNIST", "best_model", "1")


@pytest.mark.parametrize("score", [float("nan"), float("inf")])
def test_nonfinite_score_is_rejected(score):
    client = Mock()
    with pytest.raises(ValueError, match="Non-finite"):
        promote_if_better(client, "MNIST", "best_model", "2", "test/loss", score)
    client.set_registered_model_alias.assert_not_called()


def test_incompatible_selection_metadata_is_rejected():
    client = Mock()
    client.get_registered_model.return_value = SimpleNamespace(aliases={"best_model": "1"})
    client.get_model_version.return_value = SimpleNamespace(tags={})
    with pytest.raises(ValueError, match="incompatible"):
        promote_if_better(client, "MNIST", "best_model", "2", "test/loss", 1.0)
    client.set_registered_model_alias.assert_not_called()


def test_connection_errors_do_not_look_like_missing_aliases():
    client = Mock()
    client.get_registered_model.side_effect = MlflowException("unavailable")
    with pytest.raises(MlflowException, match="unavailable"):
        promote_if_better(client, "MNIST", "best_model", "2", "test/loss", 1.0)
    client.set_registered_model_alias.assert_not_called()
