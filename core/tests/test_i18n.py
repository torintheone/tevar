import pytest

from aortaplan_core import i18n


@pytest.fixture(autouse=True)
def restore_language():
    yield
    i18n.set_language(i18n.DEFAULT_LANGUAGE)


def test_default_language_is_russian():
    assert i18n.get_language() == "ru"
    assert i18n.tr("disclaimer") == "Только для исследований. Не для принятия клинических решений."


def test_all_languages_have_the_same_keys():
    reference = set(i18n.keys(i18n.DEFAULT_LANGUAGE))
    for lang in i18n.available_languages():
        assert set(i18n.keys(lang)) == reference, lang


def test_format_arguments():
    assert "SlicerVMTK" in i18n.tr("env.hint.extension", name="SlicerVMTK")
    i18n.set_language("en")
    assert i18n.tr("env.item.extension", name="X") == "Extension X"


def test_unknown_key_falls_back_to_key():
    assert i18n.tr("no.such.key") == "no.such.key"


def test_unknown_language_rejected():
    with pytest.raises(ValueError):
        i18n.set_language("xx")
