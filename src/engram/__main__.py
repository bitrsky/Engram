"""Allow running as: python -m engram"""


def _placeholder_main():
    """Placeholder until cli.py is implemented."""
    print("Engram v0.1.0")


try:
    from .cli import main
except ImportError:
    main = _placeholder_main

if __name__ == "__main__":
    main()
