# UI Restoration Design

**Date:** 2026-08-14  
**Status:** Revised after review; awaiting approval

## Purpose

Restore the compact corpus-exploration workflow lost in the schema-v1 frontend
rewrite without restoring the old scroll-synchronization machinery. The result
should read like a well-formatted spreadsheet: particle columns remain visible
in one horizontal plane, separators make each independent column easy to scan,
and examples expand within the column that owns them.

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

The `html` and `body` elements own the full-height light and dark background and
foreground colors through the Tailwind base layer. This matches the element on
which `themeManager` toggles `.dark`, covers the viewport even when content is
short, and avoids adding dark classes to every otherwise transparent child.

The header has two groups:

1. `/favicon.png` and `Natsume Simple` form the brand at the start.
2. Search controls and the theme switch form one control group at the end, with
   the theme switch last.

Responsive wrapping may move the complete control group below the brand, but it
must not distribute the theme switch into the space between brand and search.

### Autocomplete

Suggestion fetching remains debounced and independent from visibility. The
listbox is visible only while focus is anywhere within the search widget and
suggestions exist. It closes on Escape, submission, selection, or focus leaving
the complete search widget. Moving focus from the input to a suggestion button
therefore does not close the list before selection runs.
The initial populated term may be fetched but must not open the listbox on page
load. The listbox remains anchored below the input and receives a bounded height
with native vertical scrolling.

### Spreadsheet results

`ParticleOverview` is one horizontally scrollable region with `role="region"`,
an accessible label, `tabindex="0"`, and a visible focus ring. Its children are
20-rem-wide, non-shrinking particle columns in API order. At the 375-pixel smoke
viewport one complete column fits inside the page padding; at 1280 pixels roughly
three columns and part of the next remain visible, making the horizontal
continuation discoverable while giving Japanese examples 25% more width than the
current 16-rem minimum.

The page owns vertical scrolling and the region owns horizontal scrolling. The
particle headings are deliberately not sticky: `overflow-x: auto` also creates
a vertical scroll container for sticky-position containment, and bounding its
height merely to activate sticky headings would introduce an unwanted nested
vertical scroller. Native horizontal scrolling is the only scroll mechanism;
there is no synchronized header scroller, floating arrow state, resize listener,
or custom scroll-position controller.

Each column uses a normal heading, subtle vertical and horizontal separators,
and no rounded card boundary. The heading contains the particle and
returned/total count. Collocations remain independently ordered within their
particle by the server response. The same horizontal spreadsheet interaction is
intentional on mobile; it does not collapse back into a vertical card stack.

This is visually spreadsheet-like rather than an HTML table because expanded
examples have variable height. A transposed table would force an expanded cell
to increase the corresponding row height in every particle column.

### Collocations and examples

Each collocation is a full-width `<details>` element. Its `<summary>` contains
the frequency bar and noun/verb label in one aligned row. Expanded status,
errors, and example sentences render below the summary at the full column width,
without the current left indentation.

Each component assigns its example list once and never reorders or splices it,
so keyed reconciliation has no present consumer. Render it unkeyed so multiple
occurrences from one sentence remain valid. An occurrence is distinguished by
its spans, not by inventing a new public example identifier. If pagination,
reordering, or incremental loading is later added, define an occurrence key from
sentence identity plus spans before changing the list behavior.

### Theme behavior

Toggling dark mode must change both the document class and visible page colors.
No persistence or operating-system preference behavior is added; those have no
current requirement and were not part of the working behavior being restored.

## Tests

Browser coverage must demonstrate:

- brand icon and title at the start, with search and theme grouped at the end;
- the populated initial term does not open autocomplete until the input is
  focused, a suggestion remains clickable while focus moves to its button, and
  focus leaving the widget closes the listbox;
- the focusable, labelled results region overflows horizontally while all
  20-rem particle columns remain in one row at both 375- and 1280-pixel widths;
- after focusing the overflowing region, ArrowRight and End increase its
  `scrollLeft`; the same smoke also checks keyboard focus and horizontal-scroll
  discoverability after the page has been scrolled deep into a long column;
- dark-mode toggling changes computed `html`/`body` background and foreground
  colors, including when the result content is short;
- two examples sharing a sentence ID both render and no page error occurs; and
- expanded examples begin at the column edge rather than beside or indented
  under the frequency bar.

Existing keyboard, mobile-controls, safe-text rendering, corpus selection,
ranking, attribution, unit, type, and production-build checks remain protected.

## Deliberate omissions

- During each release smoke, inspect 375- and 1280-pixel viewports. Restore
  synchronized scrollers or add scroll arrows only if that evidence shows the
  native scrollbar, keyboard interaction, or touch gesture is insufficient,
  including from a deep vertical-scroll position where the bottom scrollbar is
  off screen.
- Add theme persistence only when a user preference must survive reloads.
- The next change to `ParticleColumn`'s serialized collocation key must replace
  it with explicit collocation identity and preserve open examples across a
  corpus toggle; this restoration does not need to alter that behavior.
