# Data freshness — plan

Implements FRESH-1 of [spec.md](spec.md).

## Code

`build_html()` formats the run's time as UTC + 9 hours, `%Y-%m-%d %H:%M JST`, and puts it in
`{{updated}}` in `src/template.html`'s header.

## Tests

None; it is one format string.
