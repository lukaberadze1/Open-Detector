"""Safe text extraction helpers for the Adversarial Lab's generation inputs."""

from __future__ import annotations

import ast
import io
import tokenize
import zipfile

from .documents import DocumentSection, extract_document

_MAX_ZIP_MEMBERS = 100
_MAX_ZIP_TOTAL = 20 * 1024 * 1024
_MAX_ZIP_MEMBER = 5 * 1024 * 1024
_TEXT_SUFFIXES = {".txt", ".md", ".tex", ".py", ".sh", ".bash", ".js", ".jsx", ".ts", ".tsx",
                  ".java", ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".go", ".rs", ".sql"}


def extract_generation_source(filename: str, data: bytes) -> list[DocumentSection]:
    """Extract prose or source-code comments for generation context, including bounded ZIP bundles."""
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix == ".zip":
        return _zip_sections(data)
    if suffix in {".docx", ".pptx", ".md", ".txt"}:
        return extract_document(filename, data)
    if suffix in _TEXT_SUFFIXES:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{filename} must use UTF-8 encoding.") from exc
        if suffix == ".tex":
            return [DocumentSection(filename, text, "source_text", ())] if text.strip() else []
        comments = extract_code_comments(text, suffix)
        return [DocumentSection(f"{filename}, line {line}", value, "code_comment", (line,))
                for line, value in comments if value.strip()]
    raise ValueError("Use text, Markdown, Word, PowerPoint, source-code, or ZIP files as generation context.")


def _zip_sections(data: bytes) -> list[DocumentSection]:
    stream = io.BytesIO(data)
    try:
        try:
            archive = zipfile.ZipFile(stream)
        except zipfile.BadZipFile as exc:
            raise ValueError("This is not a valid ZIP archive.") from exc
        with archive:
            infos = [item for item in archive.infolist() if not item.is_dir()]
            if len(infos) > _MAX_ZIP_MEMBERS:
                raise ValueError(f"The ZIP contains more than {_MAX_ZIP_MEMBERS} files.")
            if sum(item.file_size for item in infos) > _MAX_ZIP_TOTAL:
                raise ValueError("The ZIP expands to more than 20 MB.")
            sections: list[DocumentSection] = []
            for info in infos:
                if info.file_size > _MAX_ZIP_MEMBER:
                    raise ValueError(f"{info.filename} is larger than 5 MB.")
                name = info.filename
                suffix = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
                if suffix not in _TEXT_SUFFIXES | {".docx", ".pptx"}:
                    continue
                try:
                    content = archive.read(info)
                except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                    raise ValueError(f"Could not read {name} from the ZIP archive.") from exc
                nested = extract_generation_source(name, content)
                sections.extend(DocumentSection(f"{name} · {section.label}", section.text, section.kind,
                                                (len(sections), *section.locator)) for section in nested)
            if not sections:
                raise ValueError("The ZIP contains no supported prose or source-code comments.")
            return sections
    finally:
        stream.close()


def extract_code_comments(source: str, suffix: str) -> list[tuple[int, str]]:
    """Return comment text and line numbers without treating comment markers in strings as comments."""
    if suffix == ".py":
        comments = []
        try:
            tokens = tokenize.generate_tokens(io.StringIO(source).readline)
            for token in tokens:
                if token.type == tokenize.COMMENT:
                    comments.append((token.start[0], token.string[1:].lstrip()))
        except (tokenize.TokenError, IndentationError):
            return comments
        # Python docstrings are prose too; keep only standalone module/class/function strings.
        try:
            tree = ast.parse(source)
            nodes = [tree, *(node for node in ast.walk(tree) if isinstance(node, (ast.ClassDef, ast.FunctionDef,
                                                                                  ast.AsyncFunctionDef)))]
            for node in nodes:
                body = getattr(node, "body", [])
                if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and \
                        isinstance(body[0].value.value, str):
                    comments.append((body[0].lineno, body[0].value.value))
        except SyntaxError:
            pass
        return sorted(comments)

    # A small lexer for the common C-family, shell, and SQL comment forms. It tracks quoted strings,
    # escapes, and block comments, so comment markers embedded in literals are left alone.
    line_comment = "#" if suffix in {".sh", ".bash"} else "--" if suffix == ".sql" else "//"
    result: list[tuple[int, str]] = []
    i = 0
    line = 1
    state = "normal"
    quote = ""
    block: list[str] = []
    block_line = 0
    while i < len(source):
        char = source[i]
        nxt = source[i:i + 2]
        if char == "\n":
            line += 1
        if state == "block":
            if nxt == "*/":
                result.append((block_line, "".join(block).strip()))
                block, state, i = [], "normal", i + 2
                continue
            block.append(char)
            i += 1
            continue
        if state == "quoted":
            if char == "\\":
                i += 2
                continue
            if char == quote:
                state = "normal"
            i += 1
            continue
        if char in "\"'`":
            state, quote = "quoted", char
            i += 1
            continue
        if nxt == "/*":
            state, block_line, i = "block", line, i + 2
            continue
        if source.startswith(line_comment, i):
            end = source.find("\n", i)
            if end < 0:
                end = len(source)
            result.append((line, source[i + len(line_comment):end].strip()))
            i = end
            continue
        i += 1
    if state == "block":
        result.append((block_line, "".join(block).strip()))
    return [(number, text) for number, text in result if text]
