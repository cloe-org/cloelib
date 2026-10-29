# Contributing to cloelib

Thank you for your interest in contributing to **cloelib**! We are committed to an open science approach.

**Everyone is welcome to contribute.** You do not need to be a member of the Euclid Consortium, or of any other collaboration, to open an issue or a pull request.

> **A note for Euclid Consortium members:** to avoid duplicated effort with ongoing Consortium work, please coordinate your proposal with the KP-JC-1 coordinators before starting. This is an internal coordination step only; it does not apply to external contributors.

## Ways to Contribute

Check out [open issues](https://github.com/cloe-org/cloelib/issues) to see if development is already planned. We welcome community support on issues tagged with [`help wanted`](https://github.com/cloe-org/cloelib/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22).

If you have suggestions or found a bug, [open an issue](https://github.com/cloe-org/cloelib/issues/new) and follow the provided templates. Tag @cloe-maintainers to notify them about a possible new contribution.

## Contribution Workflow

Follow these steps to contribute:

### 1. Create a Feature Branch

```sh
git checkout -b feature/your-feature-name
```

Use descriptive branch names (e.g., `feature/add-jax-support`, `fix/convergence-issue`). Link your branch to the issue by including the issue number in the branch name or PR description, and/or leaving the branch in the associated issue in a comment.

### 2. Make Your Changes

We recommend to suggest a possible implementation in the issue description of the task and interact with the cloe-maintainers first. This will avoid unnecessary work not to be used.

- Write clear, modular code
- Add tests for new functionality
- Update documentation as needed

### 3. Commit Your Changes

```sh
git commit -m "Add feature: [brief description]"
```

Write concise, descriptive commit messages.

### 4. Push Your Branch

```sh
git push origin feature/your-feature-name
```

### 5. Open a Pull Request

Submit a pull request to the main repository. Please use the PR template and ensure you include:

- A clear description of your changes
- References to related issues
- Tests demonstrating the fix or feature

Our maintainers will review your contribution, and we'll work together to get it merged!

### Previewing the documentation of a pull request

For every pull request, the `Docs` workflow builds the documentation with `mkdocs build --strict` and uploads the resulting static site as a workflow artifact named `docs-site-pr-<PR number>`. Reviewers (and authors) can inspect the rendered pages without building them locally:

1. Open the pull request, go to the **Checks** tab and select the **Docs** workflow run (or click **Details** next to the `Docs` check).
2. Scroll to the **Artifacts** section of the run summary and download `docs-site-pr-<PR number>`.
3. Unzip it and serve it locally, for example with `python -m http.server --directory <unzipped folder> 8000`, then open <http://localhost:8000>.

The artifact is kept for 14 days and is only a preview: the published documentation is still deployed only when changes are merged into `main` or a release tag is pushed. Opening `index.html` directly with `file://` can break search and some links, so use the local server.

---

## Code Standards

We maintain high code quality using automated tools via [pre-commit](https://pre-commit.com/):

- **Linting:** [Ruff](https://docs.astral.sh/ruff/) keeps code clean
- **Formatting:** [Prettier](https://prettier.io/) and [Black](https://black.readthedocs.io/) ensure consistency
- **Type Safety:** [mypy](https://mypy.readthedocs.io/) catches type errors early

Our CI/CD pipeline also runs **unit tests** with [pytest](https://docs.pytest.org/).

All checks must pass before merging. We recommend using draft PRs if you need to iterate.

---

## Questions?

Got questions? Don't hesitate! Open an issue, [join our discussion board](https://github.com/cloe-org/cloelib/discussions), or ping @cloe-maintainers. We're here to help and love collaborating with our community.

---
