# Regulatory Notes

**This document is not legal advice.** It exists to remind you to check, not
to tell you the answer — because the answer depends on your country, and
changes over time.

## What needs checking, and when

None of this blocks development, historical data collection, backtesting,
or paper trading — all of those use only Binance's **public** market-data
endpoints, which require no account, no API key, and no identity
information. You can research and paper-trade from anywhere with internet
access.

It matters once you consider **Phase 17 (testnet)** or **Phase 18 (live
trading)** — both require a real exchange account and API keys.

## What to check before creating an exchange account

1. **Is Binance (or your chosen exchange) available in your country?**
   Exchange geo-availability changes — an exchange available today may
   restrict a country tomorrow, or vice versa. Check the exchange's own
   current terms/availability page directly before signing up.
2. **Does your country treat personal cryptocurrency trading as legal,
   restricted, or something requiring registration/reporting?** This varies
   enormously — some countries have straightforward personal-use rules,
   others require licensing for anything resembling a trading business,
   others restrict or ban it outright. Tax treatment of crypto gains also
   varies widely and is a separate question from whether trading itself is
   legal.
3. **Re-check periodically.** Regulations and exchange policies both
   change. A check done a year ago is not a check done today.

## Where this shows up in the system

`risk.yaml`'s `live_trading_gate.require_regulatory_check_ack` gates real
trading on this being acknowledged — see `docs/REAL_TRADING.md` for the
full checklist. The system cannot verify your jurisdiction's rules for you;
it can only require that you've confirmed it yourself before it will
consider unlocking testnet/live trading.

## A reasonable way to actually check

- Read the exchange's own terms of service and supported-countries page.
- Check your country's financial regulator's official guidance on
  cryptocurrency trading (search for your country + "financial regulator"
  + "cryptocurrency").
- If your situation is at all ambiguous (running this as more than a hobby,
  meaningful amounts of money, unclear local rules), talk to a professional
  who can advise on your specific situation — this document can't.

## Summary

Research and paper trading: no account needed, no restriction from this
project. Testnet and live trading: your responsibility to verify legality
and exchange availability for your situation, before you enable them.
