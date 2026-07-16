# Dual-Grok Bridge — Best Practical Design

## Why this, not browser login automation?

| Approach | Reliability | Speed to ship | Fits GRMC ethics |
|----------|-------------|---------------|------------------|
| Auto-login grok.com + drive chat UI | Low (UI churn, auth, ToS risk) | Slow | Weak |
| **File / Git courier channel** | **High** | **Fast** | **Strong** |
| Public URL fetch (httpx / Playwright) | Medium | Fast | Fine for public pages |

**Best hand:** treat the human + GitHub as the trusted bus between *web-grok* and *cli-grok*.  
Use browser tooling only for **public** pages. Never claim we can operate your logged-in Grok tab.

## Parties

- `web-grok` — conversation on grok.com  
- `cli-grok` — this environment (GRMC / local Grok Build)  
- `human` — you, the courier  
- `system` — automated notes (e.g. URL fetch)

## Files

```
bridge/
  channel_meta.json     # channel id + protocol version
  channel.jsonl         # source of truth (append log)
  active_channel.md     # human-readable mirror
  .synced_message_ids   # memory sync cursor (local, gitignored optional)
```

## Daily loop

```bash
# 1) one-time
grmc bridge init

# 2) paste what web-grok said
grmc bridge receive -t "（相手Grokの文）"
# or: grmc bridge receive -f /tmp/from_web.md

# 3) see inbox + answer as cli-grok
grmc bridge inbox
grmc bridge reply -t "（こちらの返答）"

# 4) copy into grok.com
grmc bridge paste

# 5) optional: remember in GRMC
grmc bridge sync-memory
grmc reflect
```

## Public fetch (optional)

```bash
grmc bridge fetch https://example.com -o /tmp/page.txt
grmc bridge fetch https://example.com --backend playwright   # if installed
```

Install Playwright only if you need JS rendering:

```bash
pip install playwright
playwright install chromium
```

## What we will not do

- Store your grok.com password
- Scrape private conversation DOM as the primary protocol
- Auto-write knowledge graph from bridge messages without human review

## GitHub as shared bus

Push `bridge/channel.jsonl` + `active_channel.md` when you want web-side context
in the repo (be mindful of sensitive content before pushing).
