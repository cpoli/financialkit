# Security Policy

financialkit is a scientific computing library (quantitative-finance
models, numerical algorithms and visualizations). It is educational and
research software, not investment advice, and it never fetches market
data or touches a network. It does not handle authentication,
network services, or untrusted user input in the way a web application
or server would — the realistic security surface is mainly:

- Deserializing untrusted data (e.g. loading a pickled/`.npy` object from
  an untrusted source and passing it into financialkit).
- Vulnerabilities in dependencies (numpy, scipy, matplotlib, numba).

## Reporting a Vulnerability

If you believe you've found a security vulnerability in financialkit, please
**do not open a public GitHub issue**. Instead, use GitHub's private
vulnerability reporting:

1. Go to the repository's **Security** tab.
2. Click **Report a vulnerability**.

If that's not available, open an issue asking a maintainer to contact you
privately, without describing the vulnerability itself.

We'll acknowledge reports within a few days and aim to release a fix or
mitigation promptly once a report is confirmed. Please give us reasonable
time to address the issue before any public disclosure.

## Supported Versions

Only the latest released version on PyPI is supported with security
fixes.
