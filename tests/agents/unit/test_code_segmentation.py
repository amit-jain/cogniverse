"""
Unit tests for CodeSegmentationStrategy.

Verifies tree-sitter AST parsing extracts functions, classes, methods,
and top-level blocks correctly across Python, JavaScript, TypeScript, and Go.
"""

import textwrap

import pytest

from cogniverse_runtime.ingestion.strategies import (
    CODE_EXTENSIONS,
    CodeSegmentationStrategy,
)


@pytest.fixture
def strategy():
    return CodeSegmentationStrategy(
        languages=["python", "javascript", "typescript", "go"]
    )


@pytest.fixture
def python_source(tmp_path):
    """Create a Python source file with functions, classes, and decorators."""
    code = textwrap.dedent('''\
        """Module docstring."""

        import os


        def helper(x: int) -> int:
            """Return x doubled."""
            return x * 2


        class MyClass:
            """A sample class."""

            def __init__(self, value: int):
                self.value = value

            def compute(self) -> int:
                """Compute the result."""
                return helper(self.value)

            @staticmethod
            def static_method():
                return 42


        def standalone():
            pass
    ''')
    p = tmp_path / "sample.py"
    p.write_text(code)
    return p


@pytest.fixture
def javascript_source(tmp_path):
    """Create a JavaScript source file."""
    code = textwrap.dedent('''\
        function greet(name) {
            return `Hello, ${name}!`;
        }

        class Calculator {
            add(a, b) {
                return a + b;
            }

            subtract(a, b) {
                return a - b;
            }
        }

        function multiply(a, b) {
            return a * b;
        }
    ''')
    p = tmp_path / "utils.js"
    p.write_text(code)
    return p


@pytest.fixture
def go_source(tmp_path):
    """Create a Go source file."""
    code = textwrap.dedent('''\
        package main

        import "fmt"

        type Server struct {
            Port int
            Host string
        }

        func NewServer(host string, port int) *Server {
            return &Server{Host: host, Port: port}
        }

        func (s *Server) Start() error {
            fmt.Printf("Starting server on %s:%d", s.Host, s.Port)
            return nil
        }
    ''')
    p = tmp_path / "server.go"
    p.write_text(code)
    return p


@pytest.fixture
def typescript_source(tmp_path):
    """Create a TypeScript source file."""
    code = textwrap.dedent('''\
        function validate(input: string): boolean {
            return input.length > 0;
        }

        class UserService {
            private users: string[] = [];

            addUser(name: string): void {
                this.users.push(name);
            }

            getUsers(): string[] {
                return this.users;
            }
        }
    ''')
    p = tmp_path / "service.ts"
    p.write_text(code)
    return p


class TestCodeSegmentationStrategy:
    """Unit tests for tree-sitter-based code segmentation."""

    def test_init_default_languages(self):
        s = CodeSegmentationStrategy()
        assert "python" in s.languages
        assert "typescript" in s.languages

    def test_init_custom_languages(self):
        s = CodeSegmentationStrategy(languages=["python"])
        assert s.languages == ["python"]

    def test_get_required_processors(self, strategy):
        reqs = strategy.get_required_processors()
        assert "code_file" in reqs
        assert reqs["code_file"]["languages"] == [
            "python", "javascript", "typescript", "go"
        ]

    def test_supported_extensions(self, strategy):
        exts = strategy.get_supported_extensions()
        assert ".py" in exts
        assert ".js" in exts
        assert ".ts" in exts
        assert ".go" in exts
        assert ".tsx" in exts

    def test_lang_for_extension(self):
        assert CodeSegmentationStrategy._lang_for_extension(".py") == "python"
        assert CodeSegmentationStrategy._lang_for_extension(".js") == "javascript"
        assert CodeSegmentationStrategy._lang_for_extension(".ts") == "typescript"
        assert CodeSegmentationStrategy._lang_for_extension(".go") == "go"
        assert CodeSegmentationStrategy._lang_for_extension(".rs") is None


class TestPythonParsing:
    """Test Python source file parsing."""

    def test_extracts_functions(self, strategy, python_source):
        segments = strategy.parse_file(python_source)
        names = [s["metadata"]["name"] for s in segments]
        assert "helper" in names
        assert "standalone" in names

    def test_extracts_class(self, strategy, python_source):
        segments = strategy.parse_file(python_source)
        class_segs = [s for s in segments if s["metadata"]["type"] == "class"]
        assert len(class_segs) >= 1
        assert class_segs[0]["metadata"]["name"] == "MyClass"

    def test_extracts_methods(self, strategy, python_source):
        segments = strategy.parse_file(python_source)
        names = [s["metadata"]["name"] for s in segments]
        assert "__init__" in names
        assert "compute" in names
        assert "static_method" in names

    def test_segment_metadata_has_line_numbers(self, strategy, python_source):
        segments = strategy.parse_file(python_source)
        for seg in segments:
            meta = seg["metadata"]
            assert "line_start" in meta
            assert "line_end" in meta
            assert meta["line_start"] >= 1
            assert meta["line_end"] >= meta["line_start"]

    def test_segment_metadata_has_language(self, strategy, python_source):
        segments = strategy.parse_file(python_source)
        for seg in segments:
            assert seg["metadata"]["language"] == "python"

    def test_segment_content_is_nonempty(self, strategy, python_source):
        segments = strategy.parse_file(python_source)
        for seg in segments:
            assert len(seg["content"]) > 0

    def test_signature_extraction(self, strategy, python_source):
        segments = strategy.parse_file(python_source)
        helper_seg = next(s for s in segments if s["metadata"]["name"] == "helper")
        sig = helper_seg["metadata"]["signature"]
        assert "def helper" in sig
        assert "x: int" in sig

    def test_class_signature(self, strategy, python_source):
        segments = strategy.parse_file(python_source)
        class_seg = next(
            s for s in segments
            if s["metadata"]["type"] == "class" and s["metadata"]["name"] == "MyClass"
        )
        sig = class_seg["metadata"]["signature"]
        assert "class MyClass" in sig


class TestJavaScriptParsing:
    """Test JavaScript source file parsing."""

    def test_extracts_functions(self, strategy, javascript_source):
        segments = strategy.parse_file(javascript_source)
        names = [s["metadata"]["name"] for s in segments]
        assert "greet" in names
        assert "multiply" in names

    def test_extracts_class(self, strategy, javascript_source):
        segments = strategy.parse_file(javascript_source)
        class_segs = [s for s in segments if s["metadata"]["type"] == "class"]
        assert len(class_segs) >= 1
        assert class_segs[0]["metadata"]["name"] == "Calculator"

    def test_extracts_methods(self, strategy, javascript_source):
        segments = strategy.parse_file(javascript_source)
        names = [s["metadata"]["name"] for s in segments]
        assert "add" in names
        assert "subtract" in names

    def test_language_tag(self, strategy, javascript_source):
        segments = strategy.parse_file(javascript_source)
        for seg in segments:
            assert seg["metadata"]["language"] == "javascript"


class TestGoSourceParsing:
    """Test Go source file parsing."""

    def test_extracts_functions(self, strategy, go_source):
        segments = strategy.parse_file(go_source)
        names = [s["metadata"]["name"] for s in segments]
        assert "NewServer" in names

    def test_extracts_type_declaration(self, strategy, go_source):
        segments = strategy.parse_file(go_source)
        # type_declaration maps to "class" type since it contains "type"
        # but in Go it's a struct type declaration
        names = [s["metadata"]["name"] for s in segments]
        assert "Server" in names

    def test_extracts_method(self, strategy, go_source):
        segments = strategy.parse_file(go_source)
        names = [s["metadata"]["name"] for s in segments]
        assert "Start" in names

    def test_language_tag(self, strategy, go_source):
        segments = strategy.parse_file(go_source)
        for seg in segments:
            assert seg["metadata"]["language"] == "go"


class TestTypeScriptParsing:
    """Test TypeScript source file parsing."""

    def test_extracts_functions(self, strategy, typescript_source):
        segments = strategy.parse_file(typescript_source)
        names = [s["metadata"]["name"] for s in segments]
        assert "validate" in names

    def test_extracts_class(self, strategy, typescript_source):
        segments = strategy.parse_file(typescript_source)
        class_segs = [s for s in segments if s["metadata"]["type"] == "class"]
        assert len(class_segs) >= 1
        assert class_segs[0]["metadata"]["name"] == "UserService"

    def test_extracts_methods(self, strategy, typescript_source):
        segments = strategy.parse_file(typescript_source)
        names = [s["metadata"]["name"] for s in segments]
        assert "addUser" in names
        assert "getUsers" in names


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_file(self, strategy, tmp_path):
        p = tmp_path / "empty.py"
        p.write_text("")
        segments = strategy.parse_file(p)
        assert segments == []

    def test_whitespace_only_file(self, strategy, tmp_path):
        p = tmp_path / "whitespace.py"
        p.write_text("   \n\n   \n")
        segments = strategy.parse_file(p)
        assert segments == []

    def test_imports_only_file(self, strategy, tmp_path):
        p = tmp_path / "imports_only.py"
        p.write_text("import os\nimport sys\n")
        segments = strategy.parse_file(p)
        # No functions/classes, so fallback to module segment
        assert len(segments) == 1
        assert segments[0]["metadata"]["type"] == "module"
        assert segments[0]["metadata"]["name"] == "imports_only"

    def test_unsupported_extension(self, strategy, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("a,b,c\n1,2,3\n")
        segments = strategy.parse_file(p)
        assert segments == []

    def test_language_not_in_configured(self, tmp_path):
        s = CodeSegmentationStrategy(languages=["python"])
        p = tmp_path / "code.js"
        p.write_text("function foo() {}")
        segments = s.parse_file(p)
        assert segments == []

    def test_file_with_decorators(self, strategy, tmp_path):
        code = textwrap.dedent("""\
            import functools

            def my_decorator(func):
                @functools.wraps(func)
                def wrapper(*args, **kwargs):
                    return func(*args, **kwargs)
                return wrapper

            @my_decorator
            def decorated_function(x):
                return x + 1
        """)
        p = tmp_path / "decorators.py"
        p.write_text(code)
        segments = strategy.parse_file(p)
        names = [s["metadata"]["name"] for s in segments]
        assert "my_decorator" in names
        assert "decorated_function" in names

    def test_nested_function(self, strategy, tmp_path):
        code = textwrap.dedent("""\
            def outer():
                def inner():
                    return 42
                return inner()
        """)
        p = tmp_path / "nested.py"
        p.write_text(code)
        segments = strategy.parse_file(p)
        names = [s["metadata"]["name"] for s in segments]
        assert "outer" in names

    def test_directory_walk(self, strategy, tmp_path):
        """Test that parse_file works on individual files from a directory."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("def main(): pass\n")
        (tmp_path / "src" / "util.py").write_text("def util(): pass\n")

        all_segments = []
        for f in sorted(tmp_path.rglob("*.py")):
            all_segments.extend(strategy.parse_file(f))

        names = [s["metadata"]["name"] for s in all_segments]
        assert "main" in names
        assert "util" in names


class TestCodeExtensionsMapping:
    """Test the CODE_EXTENSIONS constant."""

    def test_python_extensions(self):
        assert ".py" in CODE_EXTENSIONS["python"]

    def test_javascript_extensions(self):
        assert ".js" in CODE_EXTENSIONS["javascript"]
        assert ".jsx" in CODE_EXTENSIONS["javascript"]

    def test_typescript_extensions(self):
        assert ".ts" in CODE_EXTENSIONS["typescript"]
        assert ".tsx" in CODE_EXTENSIONS["typescript"]

    def test_go_extensions(self):
        assert ".go" in CODE_EXTENSIONS["go"]
