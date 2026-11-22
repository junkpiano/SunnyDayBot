# Favorites — spec

## Purpose

Let a reader keep the funds they hold at the top of the page, without an account.

## Behaviour

- Every card, fund or market, has a ☆ that turns to ★ when tapped.
- Starred cards also appear in a "★ Favorites" section at the top, marked "saved in this
  browser".
- Favorites are stored in this browser only.

## Acceptance

- FAV-1: Every card, fund or market, has a ☆ button that turns to ★ when starred.
- FAV-2: Starred cards are shown again in a "★ Favorites" section at the top, which is hidden
  when nothing is starred.
- FAV-3: Favorites are kept in this browser's localStorage, keyed by card title, and survive a
  reload; nothing is sent anywhere.

## Not doing

- Accounts or syncing favorites between devices.
