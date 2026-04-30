# Security Policy

## Supported Versions

Security fixes are provided for the current `main` branch and the latest published package version, when a package has been published.

## Reporting A Vulnerability

Please do not open a public issue for suspected security vulnerabilities.

Report security concerns by contacting the maintainers through the repository owner or by creating a private security advisory on GitHub when available.

Include:

- affected version or commit
- operating system and Python version
- minimal reproduction steps
- expected impact
- whether credentials, local files, Label Studio instances, or benchmark artifacts are involved

## Secrets And Local Services

Never commit real Label Studio tokens, passwords, API keys, `.env` files, SQLite runtime state, or generated workdirs. The repository ignores managed Label Studio runtime data under `docker/label_studio/data/` and internal agent artifacts under `.agents/`.

## Dependency Policy

Core runtime dependencies are intentionally small. Optional integrations such as Hugging Face, scikit-learn, datasets, and xxhash are installed through extras and should be kept isolated behind explicit optional dependency boundaries.
