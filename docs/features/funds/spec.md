# Fund coverage — spec

## Purpose

The funds SunnyDay Bot estimates, and what each one holds, so its NAV can be estimated from
market prices.

## Behaviour

Funds are shown in groups, in the order the groups first appear in the list below. Each fund is
described by:

- **NAV source:** a CSV of every NAV since inception, one `date,NAV` row per day. The fund
  company's own CSV is used where it has one (MUFG and Rakuten publish theirs by the evening);
  otherwise the Investment Trusts Association's library (投資信託協会, `toushin-lib.fwg.ne.jp`,
  by ISIN and association fund code; out late at night).
- **Legs:** the Yahoo Finance tickers it follows, with a weight. The fund's change is the
  weighted sum of the legs' changes: a weight of 2 is a 2x leveraged fund, two legs of 1 are a
  100% + 100% fund. Where an index's own daily closes are unreliable, a liquid ETF stands in
  for it (ACWI, URTH, GLDM).
- **Hedged:** hedged to yen, so USD/JPY doesn't move its NAV.
- **Tokyo ETF:** the Tokyo-listed ETF the fund holds, for a fund priced at its NAV date's own
  Tokyo close (see [estimation](../estimation/spec.md)).

| Group | Fund | Legs | Hedged | Tokyo ETF | NAV source |
|---|---|---|---|---|---|
| MSCI | eMAXIS Slim 全世界株式（オール・カントリー） | ACWI | no | — | <https://www.am.mufg.jp/fund_file/setteirai/253425.csv> |
| MSCI | 楽天・プラス・オールカントリー株式インデックス・ファンド | ACWI | no | — | <https://www.rakuten-toushin.co.jp/assets/csv/chart_100086.csv> |
| MSCI | eMAXIS Slim 全世界株式（除く日本） | ACWI | no | — | <https://www.am.mufg.jp/fund_file/setteirai/253209.csv> |
| MSCI | eMAXIS Slim 先進国株式インデックス | URTH | no | — | <https://www.am.mufg.jp/fund_file/setteirai/252653.csv> |
| S&P500 | eMAXIS Slim 米国株式（S&P500） | ^GSPC | no | — | <https://www.am.mufg.jp/fund_file/setteirai/253266.csv> |
| S&P500 | SBI・V・S&P500インデックス・ファンド | ^GSPC | no | — | <https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000J569&associFundCd=89311199> |
| S&P500 | 楽天・プラス・S&P500インデックス・ファンド | ^GSPC | no | — | <https://www.rakuten-toushin.co.jp/assets/csv/chart_100087.csv> |
| S&P500 | iFreeレバレッジ S&P500 | ^GSPC ×2 | yes | — | <https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000GUL6&associFundCd=04315188> |
| NASDAQ100 | ニッセイNASDAQ100インデックスファンド | ^NDX | no | — | <https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000PDY6&associFundCd=29313233> |
| NASDAQ100 | 楽天・プラス・NASDAQ-100インデックス・ファンド | ^NDX | no | — | <https://www.rakuten-toushin.co.jp/assets/csv/chart_100091.csv> |
| NASDAQ100 | SBI NASDAQ100インデックス・ファンド | ^NDX | no | — | <https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000SWF9&associFundCd=89311265> |
| NASDAQ100 | iFreeレバレッジ NASDAQ100 | ^NDX ×2 | yes | — | <https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000H0S5&associFundCd=0431218A> |
| NASDAQ100 | 楽天レバレッジNASDAQ-100 | ^NDX ×2 | yes | — | <https://www.rakuten-toushin.co.jp/assets/csv/chart_100071.csv> |
| NASDAQ100 | auAMレバレッジNASDAQ100為替ヘッジ無し | ^NDX ×2 | no | — | <https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000QZ77&associFundCd=AY311247> |
| NASDAQ100 | NASDAQ100 3倍ブル | ^NDX ×3 | yes | — | <https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000KW84&associFundCd=0431720A> |
| NASDAQ100 | auAMトリプル・NASDAQ100為替ヘッジ無し | ^NDX ×3 | no | — | <https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000T9H5&associFundCd=AY31226A> |
| FANG+ | iFreeNEXT FANG+インデックス | ^NYFANG | no | — | <https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000FZD4&associFundCd=04311181> |
| FANG+ | iFreeレバレッジ FANG+ | ^NYFANG ×2 | yes | — | <https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000KLQ6&associFundCd=04314208> |
| SOX | ニッセイSOX指数インデックスファンド（米国半導体株） | ^SOX | no | — | <https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000PDZ3&associFundCd=29314233> |
| SOX | 楽天・プラス・SOXインデックス・ファンド | ^SOX | no | — | <https://www.rakuten-toushin.co.jp/assets/csv/chart_100092.csv> |
| SCHD | 楽天・シュワブ・高配当株式・米国ファンド（資産成長型） | SCHD | no | — | <https://www.rakuten-toushin.co.jp/assets/csv/chart_100119.csv> |
| SCHD | 楽天・シュワブ・高配当株式・米国ファンド（四半期決算型） | SCHD | no | — | <https://www.rakuten-toushin.co.jp/assets/csv/chart_100105.csv> |
| SCHD | SBI・S・米国高配当株式ファンド（年4回決算型） | SCHD | no | — | <https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000REE2&associFundCd=8931224C> |
| Gold | 三菱UFJ 純金ファンド | GC=F | no | 1540.T | <https://www.am.mufg.jp/fund_file/setteirai/251065.csv> |
| Gold | 楽天・ゴールド・ファンド（為替ヘッジなし） | GLDM | no | — | <https://www.rakuten-toushin.co.jp/assets/csv/chart_100123.csv> |
| Gold | 楽天・ゴールド・ファンド（為替ヘッジあり） | GLDM | yes | — | <https://www.rakuten-toushin.co.jp/assets/csv/chart_100122.csv> |
| Gold | SBI・iシェアーズ・ゴールドファンド（為替ヘッジなし） | GLDM | no | — | <https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000PLM4&associFundCd=8931A236> |
| Gold | SBI・iシェアーズ・ゴールドファンド（為替ヘッジあり） | GLDM | yes | — | <https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000PLL6&associFundCd=89319236> |
| Gold | ステート・ストリート・ゴールド・オープン（為替ヘッジなし） | GLDM | no | — | <https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000RXU8&associFundCd=55311258> |
| Gold | アモーヴァ ゴールド・ファンド（為替ヘッジなし） | GLDM | no | — | <https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000F6M7&associFundCd=02312177> |
| Gold | アモーヴァ ゴールド・ファンド（為替ヘッジあり） | GLDM | yes | — | <https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000F6L9&associFundCd=02311177> |
| Gold Plus | Tracers MSCIオール・カントリー・ゴールドプラス | ACWI + GLDM | no | — | <https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000SNW3&associFundCd=02311263> |
| Gold Plus | Tracers S&P500ゴールドプラス | ^GSPC + GLDM | no | — | <https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000NS46&associFundCd=02315228> |
| Gold Plus | Tracers NASDAQ100ゴールドプラス | ^NDX + GLDM | no | — | <https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000RH60&associFundCd=02311251> |

## Acceptance

- FUND-1: Every fund in the table above has a card, in its group, in the table's order.
- FUND-2: NAV CSVs are read with both `2026/11/02` and `2026年11月02日` dates (headers may be
  Shift_JIS; data rows are ASCII); other rows are skipped.
- FUND-3: A fund whose CSV or market data can't be fetched is left off that update; the other
  funds still show.

## Not doing

- Funds outside this table: a fund is added by adding it here and to the code's fund list.
