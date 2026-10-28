import os

ROOT = "."
OUTPUT = "Structure.txt"

EXCLUDED_DIRS = {
    ".git", ".venv", "venv", "__pycache__",
    "node_modules", ".idea", ".pytest_cache",
    ".mypy_cache", "dist", "build"
}

EXCLUDED_FILES = {
    ".env", "credentials.json"
}

with open(OUTPUT, "w", encoding="utf-8") as f:
    f.write(os.path.abspath(ROOT) + "\n")

    for current, dirs, files in os.walk(ROOT):
        dirs[:] = sorted(
            d for d in dirs
            if d not in EXCLUDED_DIRS
        )

        files = sorted(
            name for name in files
            if name not in EXCLUDED_FILES
            and name != OUTPUT
        )

        depth = os.path.relpath(current, ROOT).count(os.sep) + (
            0 if current == ROOT else 1
        )

        if current == ROOT:
            depth = 0

        for name in files:
            relative = os.path.relpath(
                os.path.join(current, name), ROOT
            )
            f.write("    " * (depth + 1) + relative + "\n")

print(f"Project structure saved to {os.path.abspath(OUTPUT)}")