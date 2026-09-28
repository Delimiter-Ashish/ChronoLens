from chronolens.media import format_timestamp


def test_format_timestamp_short():
    assert format_timestamp(65.5) == "01:05.50"


def test_format_timestamp_hour():
    assert format_timestamp(3661.25) == "01:01:01.25"
