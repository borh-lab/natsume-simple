# Corpus Attribution

## Journal of Natural Language Processing

- Provider: Association for Natural Language Processing (ANLP)
- Source: https://www.anlp.jp/resource/journal_latex/
- Release used: 2026-06-15, identified and verified by the release source lock
- License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

## Japanese Wikipedia

- Provider: Wikimedia Foundation and Wikipedia contributors
- Source: `wikimedia/wikipedia`, Japanese `20231101` snapshot, with article
  identities and source revision recorded in the release source lock
- Source/legal information: https://dumps.wikimedia.org/legal.html
- License: [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/), with applicable GFDL and Wikimedia terms

## TED Talks / IWSLT 2017

- Provider: TED Conference LLC, distributed through the IWSLT 2017 Japanese-English training archive
- Source revision: `c18a4f81a47ae6fa079fe9d32db288ddde38451d`; archive identity is recorded in the release source lock
- Original source: the TED URL retained with each talk where supplied upstream
- Archive terms: the included README identifies TED copyright and CC BY-NC-ND 3.0
- Current terms: https://www.ted.com/about/our-organization/our-policies-terms/ted-com-terms-of-use
- Permission status: no license grant is asserted by this project; TED is included by owner decision while permission remains unresolved

The project converts source markup to plain text, segments and filters Japanese
sentences, normalizes Japanese lemmas, and extracts noun-particle-verb
occurrences. TED subtitles are additionally grouped into talks before sentence
filtering and splitting. The resulting database is modified material and is not
an unmodified copy of any source. No single license applies to the collection.

For attribution details, corrections, or takedown requests, contact
dev@bor.space.
