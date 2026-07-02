# Contributing

Thanks for your interest in contributing. Bug reports, fixes, features, and
documentation improvements are all welcome.

## Reporting bugs and requesting features

Please use the GitHub issue tracker. Before filing, check existing open and
recently closed issues to avoid duplicates. Useful details include:

- A reproducible test case or series of steps
- The version of the library you are using
- Any relevant modifications you have made
- Anything unusual about your environment

## Pull requests

Before sending a pull request:

1. Work against the latest source on the `main` branch.
2. Check existing and recently merged pull requests to avoid duplicating work.
3. For anything significant, open an issue first so we can discuss it before you
   invest time.

To submit:

1. Fork the repository and make your change.
2. Keep the change focused. If you also reformat unrelated code, it becomes hard
   to review what actually changed. The build runs Spotless, so run
   `mvn spotless:apply` for formatting rather than hand-reformatting.
3. Ensure the tests pass (`mvn verify`).
4. Use clear commit messages.
5. Open a pull request and respond to review feedback and CI results.

## Code of Conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md).

## Security

Please do not report security issues in public issues. See [SECURITY.md](SECURITY.md).

## Licensing

This project is licensed under Apache-2.0; see [LICENSE](LICENSE). By
contributing, you agree that your contributions will be licensed under the same
terms.
