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

## 🤝 Contributors

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind are welcome!

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://alexhall.space"><img src="https://avatars.githubusercontent.com/u/59091484?v=4?s=100" width="100px;" alt="Alex Hall"/><br /><sub><b>Alex Hall</b></sub></a><br /><a href="#bug-ahallcosmo" title="Bug reports">🐛</a> <a href="#ideas-ahallcosmo" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/itutusaus"><img src="https://avatars.githubusercontent.com/u/20775836?v=4?s=100" width="100px;" alt="itutusaus"/><br /><sub><b>itutusaus</b></sub></a><br /><a href="#review-itutusaus" title="Reviewed Pull Requests">👀</a> <a href="#projectManagement-itutusaus" title="Project Management">📆</a> <a href="#mentoring-itutusaus" title="Mentoring">🧑‍🏫</a> <a href="#promotion-itutusaus" title="Promotion">📣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/llinke1"><img src="https://avatars.githubusercontent.com/u/42432333?v=4?s=100" width="100px;" alt="Laila Linke"/><br /><sub><b>Laila Linke</b></sub></a><br /><a href="#code-llinke1" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/DavidNavarroG"><img src="https://avatars.githubusercontent.com/u/29857945?v=4?s=100" width="100px;" alt="David Navarro Gironés"/><br /><sub><b>David Navarro Gironés</b></sub></a><br /><a href="#doc-DavidNavarroG" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/stefanodavini"><img src="https://avatars.githubusercontent.com/u/206831738?v=4?s=100" width="100px;" alt="stefanodavini"/><br /><sub><b>stefanodavini</b></sub></a><br /><a href="#code-stefanodavini" title="Code">💻</a> <a href="#doc-stefanodavini" title="Documentation">📖</a> <a href="#test-stefanodavini" title="Tests">⚠️</a> <a href="#ideas-stefanodavini" title="Ideas, Planning, & Feedback">🤔</a> <a href="#tool-stefanodavini" title="Tools">🔧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://gcanasherrera.com"><img src="https://avatars.githubusercontent.com/u/13239454?v=4?s=100" width="100px;" alt="Guadalupe Cañas-Herrera"/><br /><sub><b>Guadalupe Cañas-Herrera</b></sub></a><br /><a href="#code-gcanasherrera" title="Code">💻</a> <a href="#maintenance-gcanasherrera" title="Maintenance">🚧</a> <a href="#ideas-gcanasherrera" title="Ideas, Planning, & Feedback">🤔</a> <a href="#bug-gcanasherrera" title="Bug reports">🐛</a> <a href="#content-gcanasherrera" title="Content">🖋</a> <a href="#data-gcanasherrera" title="Data">🔣</a> <a href="#doc-gcanasherrera" title="Documentation">📖</a> <a href="#infra-gcanasherrera" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#projectManagement-gcanasherrera" title="Project Management">📆</a> <a href="#question-gcanasherrera" title="Answering Questions">💬</a> <a href="#test-gcanasherrera" title="Tests">⚠️</a> <a href="#talk-gcanasherrera" title="Talks">📢</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://marcobonici.github.io/"><img src="https://avatars.githubusercontent.com/u/58727599?v=4?s=100" width="100px;" alt="Marco Bonici"/><br /><sub><b>Marco Bonici</b></sub></a><br /><a href="#code-marcobonici" title="Code">💻</a> <a href="#maintenance-marcobonici" title="Maintenance">🚧</a> <a href="#ideas-marcobonici" title="Ideas, Planning, & Feedback">🤔</a> <a href="#bug-marcobonici" title="Bug reports">🐛</a> <a href="#content-marcobonici" title="Content">🖋</a> <a href="#doc-marcobonici" title="Documentation">📖</a> <a href="#infra-marcobonici" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#projectManagement-marcobonici" title="Project Management">📆</a> <a href="#question-marcobonici" title="Answering Questions">💬</a> <a href="#test-marcobonici" title="Tests">⚠️</a> <a href="#talk-marcobonici" title="Talks">📢</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/chiaramoretti"><img src="https://avatars.githubusercontent.com/u/12472732?v=4?s=100" width="100px;" alt="Chiara Moretti"/><br /><sub><b>Chiara Moretti</b></sub></a><br /><a href="#code-chiaramoretti" title="Code">💻</a> <a href="#maintenance-chiaramoretti" title="Maintenance">🚧</a> <a href="#ideas-chiaramoretti" title="Ideas, Planning, & Feedback">🤔</a> <a href="#bug-chiaramoretti" title="Bug reports">🐛</a> <a href="#content-chiaramoretti" title="Content">🖋</a> <a href="#doc-chiaramoretti" title="Documentation">📖</a> <a href="#talk-chiaramoretti" title="Talks">📢</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/AndreaPezzotta"><img src="https://avatars.githubusercontent.com/u/29603598?v=4?s=100" width="100px;" alt="AndreaPezzotta"/><br /><sub><b>AndreaPezzotta</b></sub></a><br /><a href="#code-AndreaPezzotta" title="Code">💻</a> <a href="#maintenance-AndreaPezzotta" title="Maintenance">🚧</a> <a href="#ideas-AndreaPezzotta" title="Ideas, Planning, & Feedback">🤔</a> <a href="#bug-AndreaPezzotta" title="Bug reports">🐛</a> <a href="#content-AndreaPezzotta" title="Content">🖋</a> <a href="#data-AndreaPezzotta" title="Data">🔣</a> <a href="#doc-AndreaPezzotta" title="Documentation">📖</a> <a href="#talk-AndreaPezzotta" title="Talks">📢</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.cosmostat.org/people/santiago-casas"><img src="https://avatars.githubusercontent.com/u/6987716?v=4?s=100" width="100px;" alt="Santiago Casas"/><br /><sub><b>Santiago Casas</b></sub></a><br /><a href="#code-santiagocasas" title="Code">💻</a> <a href="#maintenance-santiagocasas" title="Maintenance">🚧</a> <a href="#ideas-santiagocasas" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/PedroCarrilho"><img src="https://avatars.githubusercontent.com/u/60090062?v=4?s=100" width="100px;" alt="Pedro Carrilho"/><br /><sub><b>Pedro Carrilho</b></sub></a><br /><a href="#code-PedroCarrilho" title="Code">💻</a> <a href="#maintenance-PedroCarrilho" title="Maintenance">🚧</a> <a href="#ideas-PedroCarrilho" title="Ideas, Planning, & Feedback">🤔</a> <a href="#bug-PedroCarrilho" title="Bug reports">🐛</a> <a href="#content-PedroCarrilho" title="Content">🖋</a> <a href="#data-PedroCarrilho" title="Data">🔣</a> <a href="#doc-PedroCarrilho" title="Documentation">📖</a> <a href="#talk-PedroCarrilho" title="Talks">📢</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://ntessore.page"><img src="https://avatars.githubusercontent.com/u/3993688?v=4?s=100" width="100px;" alt="Nicolas Tessore"/><br /><sub><b>Nicolas Tessore</b></sub></a><br /><a href="#tool-ntessore" title="Tools">🔧</a> <a href="#mentoring-ntessore" title="Mentoring">🧑‍🏫</a> <a href="#code-ntessore" title="Code">💻</a> <a href="#review-ntessore" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://sfarrens.github.io"><img src="https://avatars.githubusercontent.com/u/6851839?v=4?s=100" width="100px;" alt="Samuel Farrens"/><br /><sub><b>Samuel Farrens</b></sub></a><br /><a href="#tool-sfarrens" title="Tools">🔧</a> <a href="#mentoring-sfarrens" title="Mentoring">🧑‍🏫</a> <a href="#code-sfarrens" title="Code">💻</a> <a href="#review-sfarrens" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/josecolomanadal"><img src="https://avatars.githubusercontent.com/u/83759085?v=4?s=100" width="100px;" alt="Jose Coloma Nadal"/><br /><sub><b>Jose Coloma Nadal</b></sub></a><br /><a href="#bug-josecolomanadal" title="Bug reports">🐛</a> <a href="#code-josecolomanadal" title="Code">💻</a> <a href="#ideas-josecolomanadal" title="Ideas, Planning, & Feedback">🤔</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/caspervedder"><img src="https://avatars.githubusercontent.com/u/187176614?v=4?s=100" width="100px;" alt="Casper Vedder"/><br /><sub><b>Casper Vedder</b></sub></a><br /><a href="#code-caspervedder" title="Code">💻</a> <a href="#ideas-caspervedder" title="Ideas, Planning, & Feedback">🤔</a> <a href="#bug-caspervedder" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://benbose.com/"><img src="https://avatars.githubusercontent.com/u/45853389?v=4?s=100" width="100px;" alt="Ben Bose"/><br /><sub><b>Ben Bose</b></sub></a><br /><a href="#code-nebblu" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/didamarkovic"><img src="https://avatars.githubusercontent.com/u/9950748?v=4?s=100" width="100px;" alt="Dida Markovic"/><br /><sub><b>Dida Markovic</b></sub></a><br /><a href="#ideas-didamarkovic" title="Ideas, Planning, & Feedback">🤔</a> <a href="#question-didamarkovic" title="Answering Questions">💬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/m-aguena"><img src="https://avatars.githubusercontent.com/u/12038660?v=4?s=100" width="100px;" alt="Michel Aguena"/><br /><sub><b>Michel Aguena</b></sub></a><br /><a href="#code-m-aguena" title="Code">💻</a> <a href="#test-m-aguena" title="Tests">⚠️</a> <a href="#ideas-m-aguena" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ippoppi"><img src="https://avatars.githubusercontent.com/u/50492103?v=4?s=100" width="100px;" alt="Filippo Oppizzi"/><br /><sub><b>Filippo Oppizzi</b></sub></a><br /><a href="#code-ippoppi" title="Code">💻</a> <a href="#ideas-ippoppi" title="Ideas, Planning, & Feedback">🤔</a> <a href="#tool-ippoppi" title="Tools">🔧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://jaimeruizzapatero.net/"><img src="https://avatars.githubusercontent.com/u/39957598?v=4?s=100" width="100px;" alt="Jaime RZ"/><br /><sub><b>Jaime RZ</b></sub></a><br /><a href="#ideas-JaimeRZP" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://arthurmloureiro.github.io"><img src="https://avatars.githubusercontent.com/u/6471279?v=4?s=100" width="100px;" alt="Arthur Loureiro"/><br /><sub><b>Arthur Loureiro</b></sub></a><br /><a href="#code-arthurmloureiro" title="Code">💻</a> <a href="#ideas-arthurmloureiro" title="Ideas, Planning, & Feedback">🤔</a> <a href="#doc-arthurmloureiro" title="Documentation">📖</a> <a href="#review-arthurmloureiro" title="Reviewed Pull Requests">👀</a> <a href="#bug-arthurmloureiro" title="Bug reports">🐛</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/pburger112"><img src="https://avatars.githubusercontent.com/u/51719634?v=4?s=100" width="100px;" alt="Pierre Burger"/><br /><sub><b>Pierre Burger</b></sub></a><br /><a href="#code-pburger112" title="Code">💻</a> <a href="#ideas-pburger112" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/davidesciotti"><img src="https://avatars.githubusercontent.com/u/84071067?v=4?s=100" width="100px;" alt="Davide Sciotti"/><br /><sub><b>Davide Sciotti</b></sub></a><br /><a href="#bug-davidesciotti" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/GabrieleParimbelli"><img src="https://avatars.githubusercontent.com/u/43963112?v=4?s=100" width="100px;" alt="GabrieleParimbelli"/><br /><sub><b>GabrieleParimbelli</b></sub></a><br /><a href="#bug-GabrieleParimbelli" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/zahrabaghkhani"><img src="https://avatars.githubusercontent.com/u/47903409?v=4?s=100" width="100px;" alt="Zahra Baghkhani"/><br /><sub><b>Zahra Baghkhani</b></sub></a><br /><a href="#code-zahrabaghkhani" title="Code">💻</a> <a href="#ideas-zahrabaghkhani" title="Ideas, Planning, & Feedback">🤔</a> <a href="#bug-zahrabaghkhani" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/fabriceroy"><img src="https://avatars.githubusercontent.com/u/29073232?v=4?s=100" width="100px;" alt="Fabrice Roy"/><br /><sub><b>Fabrice Roy</b></sub></a><br /><a href="#doc-fabriceroy" title="Documentation">📖</a> <a href="#code-fabriceroy" title="Code">💻</a> <a href="#ideas-fabriceroy" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/pltaylor16"><img src="https://avatars.githubusercontent.com/u/22646673?v=4?s=100" width="100px;" alt="pltaylor16"/><br /><sub><b>pltaylor16</b></sub></a><br /><a href="#code-pltaylor16" title="Code">💻</a> <a href="#doc-pltaylor16" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/rreischke"><img src="https://avatars.githubusercontent.com/u/31727230?v=4?s=100" width="100px;" alt="Robert Reischke"/><br /><sub><b>Robert Reischke</b></sub></a><br /><a href="#ideas-rreischke" title="Ideas, Planning, & Feedback">🤔</a> <a href="#mentoring-rreischke" title="Mentoring">🧑‍🏫</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/AngusWright"><img src="https://avatars.githubusercontent.com/u/5625880?v=4?s=100" width="100px;" alt="Angus H. Wright"/><br /><sub><b>Angus H. Wright</b></sub></a><br /><a href="#ideas-AngusWright" title="Ideas, Planning, & Feedback">🤔</a> <a href="#mentoring-AngusWright" title="Mentoring">🧑‍🏫</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/matteobaratto"><img src="https://avatars.githubusercontent.com/u/75221958?v=4?s=100" width="100px;" alt="Matteo Baratto "/><br /><sub><b>Matteo Baratto </b></sub></a><br /><a href="#code-matteobaratto" title="Code">💻</a> <a href="#ideas-matteobaratto" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/MariaTsedrik"><img src="https://avatars.githubusercontent.com/u/93711395?v=4?s=100" width="100px;" alt="Maria Tsedrik"/><br /><sub><b>Maria Tsedrik</b></sub></a><br /><a href="#code-MariaTsedrik" title="Code">💻</a> <a href="#ideas-MariaTsedrik" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/arsouki"><img src="https://avatars.githubusercontent.com/u/162714090?v=4?s=100" width="100px;" alt="Arghavan Souki"/><br /><sub><b>Arghavan Souki</b></sub></a><br /><a href="#bug-arsouki" title="Bug reports">🐛</a> <a href="#doc-arsouki" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/zsirap"><img src="https://avatars.githubusercontent.com/u/50758399?v=4?s=100" width="100px;" alt="zsirap"/><br /><sub><b>zsirap</b></sub></a><br /><a href="#code-zsirap" title="Code">💻</a> <a href="#ideas-zsirap" title="Ideas, Planning, & Feedback">🤔</a> <a href="#bug-zsirap" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ktanidis"><img src="https://avatars.githubusercontent.com/u/60473500?v=4?s=100" width="100px;" alt="Konstantinos Tanidis"/><br /><sub><b>Konstantinos Tanidis</b></sub></a><br /><a href="#code-ktanidis" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://chaitanyachawak.github.io/"><img src="https://avatars.githubusercontent.com/u/55046588?v=4?s=100" width="100px;" alt="Chaitanya"/><br /><sub><b>Chaitanya</b></sub></a><br /><a href="#code-ChaitanyaChawak" title="Code">💻</a> <a href="#bug-ChaitanyaChawak" title="Bug reports">🐛</a> <a href="#review-ChaitanyaChawak" title="Reviewed Pull Requests">👀</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/raphkou"><img src="https://avatars.githubusercontent.com/u/61792335?v=4?s=100" width="100px;" alt="raphkou"/><br /><sub><b>raphkou</b></sub></a><br /><a href="#bug-raphkou" title="Bug reports">🐛</a> <a href="#code-raphkou" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ivansladoljev"><img src="https://avatars.githubusercontent.com/u/144113061?v=4?s=100" width="100px;" alt="Ivan Sladoljev"/><br /><sub><b>Ivan Sladoljev</b></sub></a><br /><a href="#code-ivansladoljev" title="Code">💻</a> <a href="#ideas-ivansladoljev" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/NastassiaG"><img src="https://avatars.githubusercontent.com/u/107264848?v=4?s=100" width="100px;" alt="NastassiaG"/><br /><sub><b>NastassiaG</b></sub></a><br /><a href="#code-NastassiaG" title="Code">💻</a> <a href="#bug-NastassiaG" title="Bug reports">🐛</a> <a href="#ideas-NastassiaG" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/martincrocce"><img src="https://avatars.githubusercontent.com/u/29067049?v=4?s=100" width="100px;" alt="Martin Crocce"/><br /><sub><b>Martin Crocce</b></sub></a><br /><a href="#projectManagement-martincrocce" title="Project Management">📆</a> <a href="#mentoring-martincrocce" title="Mentoring">🧑‍🏫</a> <a href="#promotion-martincrocce" title="Promotion">📣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/CarmelitaCarbone"><img src="https://avatars.githubusercontent.com/u/17458225?v=4?s=100" width="100px;" alt="CarmelitaCarbone"/><br /><sub><b>CarmelitaCarbone</b></sub></a><br /><a href="#projectManagement-CarmelitaCarbone" title="Project Management">📆</a> <a href="#mentoring-CarmelitaCarbone" title="Mentoring">🧑‍🏫</a> <a href="#promotion-CarmelitaCarbone" title="Promotion">📣</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
