# Data freshness — plan

Implements FRESH-1 of [spec.md](spec.md).

## Code

`build_html()` formats the run's time as UTC + 9 hours, `%Y-%m-%d %H:%M JST`, and puts it in
`{{updated}}` in `src/template.html`'s header.

## Tests

- `shows_funds_in_their_groups` checks the header's `Updated YYYY-MM-DD HH:MM JST` (FRESH-1).
