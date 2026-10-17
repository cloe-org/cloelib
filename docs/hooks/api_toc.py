"""MkDocs hook: keep only packages and modules in the API page's table of contents.

Classes, functions and other members are still rendered on the page; they are
only removed from the sidebar. The heading levels in api.md can therefore
follow the package structure directly.
"""

from pathlib import Path

PACKAGE_ROOT = (
    Path(__file__).resolve().parents[2]
)  # repository root (this file lives in docs/hooks/)
API_PAGES = {"api.md"}


def _keep(anchor: str) -> bool:
    """Keep ordinary headings, and API headings only for packages/modules."""
    if not anchor.startswith("cloelib"):
        return True
    path = PACKAGE_ROOT.joinpath(*anchor.split("."))
    return path.is_dir() or path.with_suffix(".py").is_file()


def _prune(items):
    kept = []
    for item in items:
        item.children = _prune(item.children)
        if _keep(item.id):
            kept.append(item)
    return kept


def on_page_content(html, page, **kwargs):
    if page.file.src_uri in API_PAGES:
        page.toc.items = _prune(page.toc.items)
    return html
