# Favorites — plan

Implements FAV-1 to FAV-3 of [spec.md](spec.md).

## Code

All in the script in `src/template.html`:

- On load, a star button is prepended to every card's title, with `data-name` set to the title.
- `favs` is a `Set` loaded from `localStorage['favorites']` (a JSON array of titles).
- `render()` fills the "★ Favorites" section with copies of the starred cards (without an open
  chart), hides the section when it is empty, and sets every star's ☆/★, title and
  `aria-pressed`.
- The delegated click handler toggles a star's title in `favs`, saves it, and calls `render()`.

## Tests

None in CI: the page script has no test runner here. The 2026-11-22 browser check confirmed that starring a card doesn't close its chart,
and that the favorites copy has no chart.

## Limits

- Keyed by card title: renaming a fund drops it from favorites.
