# UI Restoration Design

**Date:** 2026-08-14  
**Status:** Approved for implementation planning

## Purpose

Restore the compact corpus-exploration workflow lost in the schema-v1 frontend
rewrite without restoring the old scroll-synchronization machinery. The result
should read like a well-formatted spreadsheet: particle columns remain visible
in one horizontal plane, rows align visually, and examples expand within the
column that owns them.

## Confirmed regressions

- The schema-v1 rewrite replaced the horizontal particle view with a responsive
  card grid. Later particles now require unrelated vertical scrolling.
- Suggestions open when an asynchronous result arrives even when the search
  input is not focused. The initial search term therefore opens the list on page
  load.
- Applying `.dark` changes selected controls but no longer changes the page-level
  background or foreground. The former page shell owned those colors.
- `時間を判断する` returns two occurrences in sentence `74762`. Examples are
  keyed only by sentence ID, so Svelte raises `each_key_duplicate` and leaves the
  disclosure displaying `Loading examples…`.
- The frequency bar is a sibling of the complete `<details>` element. Expanded
  examples consequently use only the narrow space remaining beside the bar and
  add another left margin.
- The schema-v1 header dropped the existing `/favicon.png` brand mark.

## Design

### Page shell and header

The page owns its full-height light and dark background and foreground colors.
This restores a single theme boundary instead of adding dark classes to every
otherwise transparent child.

The header has two groups:

1. `/favicon.png` and `Natsume Simple` form the brand at the start.
2. Search controls and the theme switch form one control group at the end, with
   the theme switch last.

Responsive wrapping may move the complete control group below the brand, but it
must not distribute the theme switch into the space between brand and search.

### Autocomplete

Suggestion fetching remains debounced and independent from visibility. The
listbox is visible only while the search input owns focus and suggestions exist.
It closes on Escape, submission, selection, or focus leaving the search widget.
The initial populated term may be fetched but must not open the listbox on page
load. The listbox remains anchored below the input and receives a bounded height
with native vertical scrolling.

### Spreadsheet results

`ParticleOverview` is one labelled, horizontally scrollable region. Its children
are fixed-width, non-shrinking particle columns in API order. Native horizontal
scrolling is the only scroll mechanism; there is no synchronized header scroller,
floating arrow state, resize listener, or custom scroll-position controller.

Each column uses a sticky heading within the results region, subtle vertical and
horizontal separators, and no rounded card boundary. The heading contains the
particle and returned/total count. Collocations remain independently ordered
within their particle by the server response.

This is visually spreadsheet-like rather than an HTML table because expanded
examples have variable height. A transposed table would force an expanded cell
to increase the corresponding row height in every particle column.

### Collocations and examples

Each collocation is a full-width `<details>` element. Its `<summary>` contains
the frequency bar and noun/verb label in one aligned row. Expanded status,
errors, and example sentences render below the summary at the full column width,
without the current left indentation.

Example responses are immutable lists and do not require keyed reconciliation.
Render them unkeyed so multiple occurrences from one sentence remain valid. An
occurrence is distinguished by its spans, not by inventing a new public example
identifier.

### Theme behavior

Toggling dark mode must change both the document class and visible page colors.
No persistence or operating-system preference behavior is added; those have no
current requirement and were not part of the working behavior being restored.

## Tests

Browser coverage must demonstrate:

- brand icon and title at the start, with search and theme grouped at the end;
- the populated initial term does not open autocomplete until the input is
  focused, and focus leaving the widget closes it;
- the results region overflows horizontally while all particle columns remain
  in one row;
- dark-mode toggling changes computed page background and foreground colors;
- two examples sharing a sentence ID both render and no page error occurs; and
- expanded examples begin at the column edge rather than beside or indented
  under the frequency bar.

Existing keyboard, mobile-controls, safe-text rendering, corpus selection,
ranking, attribution, unit, type, and production-build checks remain protected.

## Deliberate omissions

- Restore synchronized header/body scrollers only if native horizontal scrolling
  proves insufficient in user testing.
- Add scroll arrows only if users cannot discover the native scrollbar or touch
  gesture.
- Add theme persistence only when a user preference must survive reloads.

