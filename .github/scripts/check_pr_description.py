"""Check that a pull-request description has the items required by the PR template.

The PR body is read from the ``PR_BODY`` environment variable. The script reports
every missing item as a GitHub error annotation and exits with status 1 if any
item is missing. It only checks that the items are present, not that they are
true: understanding of the code remains the author's responsibility.

Required items (see ``.github/pull_request_template.md``):

* a non-empty "Summary" section,
* a non-empty "AI Assistance" section (write "None." if no AI was used),
* the checked box "I have reviewed and understand ...",
* the checked box "I have filled in the AI Assistance section ...".
"""

from __future__ import annotations

import os
import re
import sys

HEADING = re.compile(r"^#{1,6}\s+(?P<title>.*?)\s*$", re.MULTILINE)
COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def get_section(body: str, name: str) -> str | None:
    """Return the text of the section whose heading ends with ``name``.

    The text is taken up to the next heading, with HTML comments removed.
    Returns ``None`` if the section heading is not found.
    """
    headings = list(HEADING.finditer(body))
    for index, match in enumerate(headings):
        if match.group("title").lower().endswith(name.lower()):
            end = (
                headings[index + 1].start() if index + 1 < len(headings) else len(body)
            )
            return COMMENT.sub("", body[match.end() : end]).strip()
    return None


def is_checked(body: str, text: str) -> bool:
    """Return True if a checked task-list item starting with ``text`` exists."""
    pattern = rf"^\s*[-*]\s+\[[xX]\]\s+{re.escape(text)}"
    return re.search(pattern, COMMENT.sub("", body), re.MULTILINE) is not None


def find_problems(body: str) -> list[str]:
    """Return one message per required item missing from the PR description."""
    problems = []
    if not get_section(body, "Summary"):
        problems.append(
            "The 'Summary' section is missing or empty. Describe in your own words what this PR does."
        )
    if not get_section(body, "AI Assistance"):
        problems.append(
            "The 'AI Assistance' section is missing or empty. Describe how AI was used, or write 'None.'"
        )
    if not is_checked(body, "I have reviewed and understand"):
        problems.append(
            "The checkbox 'I have reviewed and understand every change in this PR ...' is missing or not checked."
        )
    if not is_checked(body, "I have filled in the AI Assistance section"):
        problems.append(
            "The checkbox 'I have filled in the AI Assistance section ...' is missing or not checked."
        )
    return problems


def main() -> int:
    body = os.environ.get("PR_BODY") or ""
    problems = find_problems(body)
    for problem in problems:
        print(f"::error title=Incomplete PR description::{problem}")
    if problems:
        print(
            "\nPlease complete the PR description using .github/pull_request_template.md "
            "and edit the PR; this check re-runs automatically when the description changes."
        )
        return 1
    print("PR description contains all required items.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
