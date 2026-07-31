import pytest

from discovery.model import Model, parse_blocks

SOURCE = """\
from numpyro import sample
from numpyro.distributions import Normal


def model(x, y=None):
    # @discovery:begin prior
    slope = sample("slope", Normal(0.0, 1.0))
    # @discovery:end
    mu = slope * x
    # @discovery:begin likelihood
    return sample("y", Normal(mu, 1.0), obs=y)
    # @discovery:end
"""


def test_blocks_are_parsed_in_order_with_indentation():
    blocks = Model(SOURCE).blocks
    assert [block.name for block in blocks] == ["prior", "likelihood"]
    assert blocks[0].body == '    slope = sample("slope", Normal(0.0, 1.0))'
    assert blocks[0].indent == "    "


def test_unnamed_blocks_get_positional_names():
    source = "# @discovery:begin\nx = 1\n# @discovery:end\n"
    assert [block.name for block in Model(source).blocks] == ["block_0"]


@pytest.mark.parametrize(
    "source",
    [
        "# @discovery:begin a\nx = 1\n",
        "# @discovery:end\n",
        "# @discovery:begin a\n# @discovery:begin b\n# @discovery:end\n",
    ],
)
def test_malformed_markers_are_rejected(source):
    with pytest.raises(ValueError):
        Model(source).blocks


def test_substitute_replaces_only_the_named_block():
    new = Model(SOURCE).substitute(
        {"prior": '    slope = sample("slope", Cauchy(0.0, 1.0))'}
    )
    assert 'Cauchy(0.0, 1.0))' in new.source
    assert 'Normal(0.0, 1.0))' not in new.source
    # scaffolding and the other block survive untouched
    assert "mu = slope * x" in new.source
    assert '    return sample("y", Normal(mu, 1.0), obs=y)' in new.source
    # the result is still a well formed model with the same blocks
    assert [b.name for b in new.blocks] == ["prior", "likelihood"]


def test_substitute_handles_multi_line_bodies():
    new = Model(SOURCE).substitute({"prior": "    a = 1\n    b = 2\n    slope = a + b"})
    assert new.blocks[0].body == "    a = 1\n    b = 2\n    slope = a + b"


def test_substitute_indents_a_body_written_flush_left():
    # what a model actually returns: the region on its own, at column zero
    new = Model(SOURCE).substitute({"prior": 'slope = sample("slope", Cauchy(0.0, 1.0))'})
    assert '    slope = sample("slope", Cauchy(0.0, 1.0))' in new.source
    assert compile(new.source, "<test>", "exec")


def test_substitute_keeps_shape_within_an_indented_body():
    new = Model(SOURCE).substitute({"prior": "if True:\n    slope = 1.0\n\nelse:\n    slope = 2.0"})
    assert new.blocks[0].body == "    if True:\n        slope = 1.0\n\n    else:\n        slope = 2.0"
    assert compile(new.source, "<test>", "exec")


def test_substitute_rejects_unknown_blocks():
    with pytest.raises(KeyError, match="nonsense"):
        Model(SOURCE).substitute({"nonsense": "x = 1"})


def test_original_model_is_unchanged():
    model = Model(SOURCE)
    model.substitute({"prior": "    slope = 0.0"})
    assert model.source == SOURCE


def test_parse_blocks_reads_a_fenced_reply():
    reply = (
        "Sure, here is a heavier tailed prior.\n\n"
        "```python\n"
        "    # @discovery:begin prior\n"
        '    slope = sample("slope", StudentT(3.0, 0.0, 1.0))\n'
        "    # @discovery:end\n"
        "```\n"
        "This should be more robust to outliers.\n"
    )
    assert parse_blocks(reply) == {
        "prior": '    slope = sample("slope", StudentT(3.0, 0.0, 1.0))'
    }


def test_parse_blocks_reads_an_unfenced_reply():
    reply = "# @discovery:begin prior\nslope = 0.0\n# @discovery:end"
    assert parse_blocks(reply) == {"prior": "slope = 0.0"}


def test_parse_blocks_ignores_prose_without_markers():
    assert parse_blocks("I could not think of anything better.") == {}
