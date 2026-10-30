"""Command-line access to the shared representation."""
import sys


def main(argv=None):
    from .core.cli import main as execute
    argv = list(sys.argv[1:] if argv is None else argv)
    return execute(argv[1:] if argv[:1] == ["core"] else argv)
