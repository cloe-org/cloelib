# Contributing to cloelib

Thank you for your interest in contributing to **cloelib**! We are doing an effor to keep an open science approach

## Ways to Contribute

Check out [open issues](https://github.com/cloe-org/cloelib/issues) to see if development is already planned. We welcome community support on issues tagged with [`help wanted`](https://github.com/cloe-org/cloelib/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22).

If you have suggestions or found a bug, [open an issue](https://github.com/cloe-org/cloelib/issues/new) and follow the provided templates. Tag @cloe-maintainers to notify them about a possible new contribution.

## Contribution Workflow

Follow these steps to contribute:

### 1️⃣ Create a Feature Branch

```sh
git checkout -b feature/your-feature-name
```

Use descriptive branch names (e.g., `feature/add-jax-support`, `fix/convergence-issue`). Link your branch to the issue by including the issue number in the branch name or PR description, and/or leaving the branch in the associated issue in a comment.

### 2️⃣ Make Your Changes

We recommend to suggest a possible implementation in the issue description of the task and interact with the cloe-maintainers first. This will avoid unnecessary work not to be used.

- Write clear, modular code
- Add tests for new functionality
- Update documentation as needed

### 3️⃣ Commit Your Changes

```sh
git commit -m "Add feature: [brief description]"
```

Write concise, descriptive commit messages.

### 4️⃣ Push Your Branch

```sh
git push origin feature/your-feature-name
```

### 5️⃣ Open a Pull Request

Submit a pull request to the main repository. Please use the PR template and ensure you include:

- A clear description of your changes
- References to related issues
- Tests demonstrating the fix or feature

Our maintainers will review your contribution, and we'll work together to get it merged!

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

--8<-- "README.md:contributors"
