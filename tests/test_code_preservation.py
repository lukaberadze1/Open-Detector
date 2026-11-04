"""Source input extraction includes comment scopes while leaving code and literals alone."""

from opendetector.generator_inputs import extract_code_comments, extract_generation_source


def test_python_comments_and_docstrings_are_extracted_without_string_literals():
    source = '''"""Module docs mention delve into this."""
MESSAGE = "delve into this; # still a string"
URL = "https://example.test/path"
# A real comment.
def run(value):
    """Function documentation."""
    result = value + 1  # inline note
    return result
'''
    original_lines = source.splitlines()
    comments = extract_code_comments(source, ".py")

    assert comments == [
        (1, "Module docs mention delve into this."),
        (4, "A real comment."),
        (6, "Function documentation."),
        (7, "inline note"),
    ]
    # The parser returns comment text only; executable statements and literal contents never enter its output.
    assert not any("MESSAGE" in comment or "https://" in comment or "result =" in comment
                   for _, comment in comments)
    assert original_lines[1] == 'MESSAGE = "delve into this; # still a string"'


def test_c_family_comment_markers_inside_strings_are_not_comments():
    source = '''const phrase = "delve into this // not a comment";
const url = "https://example.test";
/* A block comment with a trigger phrase. */
int count = 2; // actual trailing comment
'''
    assert extract_code_comments(source, ".js") == [
        (3, "A block comment with a trigger phrase."),
        (4, "actual trailing comment"),
    ]


def test_generator_context_for_code_contains_only_comments():
    source = 'const prompt = "delve into this";\n// Explain the next step.\nrun(prompt);\n'
    sections = extract_generation_source("example.js", source.encode())
    assert [section.text for section in sections] == ["Explain the next step."]
