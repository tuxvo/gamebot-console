# 🎮 Gamebot: A Retro AI Game Console for Discord

One Discord bot, three swappable "cartridges" — all playable with **zero typing**:

| Cartridge | What it does | Powered by |
|---|---|---|
| ♟️ **Emoji Chess** | Full legal chess (castling, en passant, promotions, clocks, every draw rule) played entirely with buttons and reaction taps | python-chess |
| ♿ **AI Accessibility Assist** | Type your *intent* in plain English (`!aiassist save my knight`) — the AI scans the real legal-move list, highlights ONE piece with a single 🔴, and one tap plays it. Built for mobile users and players with motor difficulties. | gpt-4o-mini via Microsoft GitHub Models on Azure AI (drop-in support for Azure OpenAI deployments) |
| 🗡️ **Infinite RPG** + ❓ **Trivia Master** | Endless button-driven adventure and instant 4-button trivia on any topic, using strict Structured JSON Outputs — no chat window, ever | Gemini 2.5 Flash (google-genai) |

## Setup

```bash
pip install -U discord.py python-chess google-genai openai
```

1. Open `chessfinal.py` and fill in the CONFIG section at the top:
   - `bot.run("...")` (bottom of file) — your Discord bot token
   - `GEMINI_API_KEY` — from Google AI Studio (AIza...) or Vertex express mode (AQ...) — both auto-detected
   - `GITHUB_TOKEN` — a fine-grained GitHub token with the **Models: Read** permission (free), **or** fill the `AZURE_OPENAI_*` lines with a real Azure OpenAI deployment (used first when present)
2. Run it:

```bash
python chessfinal.py
```

3. In Discord, type `!help`.

## Commands

`!startchess [minutes]` · `!resign [white/black]` · `!offerdraw` / `!acceptdraw` · `!claimdraw` · `!aiassist <goal>` · `!startrpg` · `!starttrivia <topic>` · `!help`

## Architecture notes

- **Nothing illegal can ever be offered:** every color circle, direction arrow, step number and AI suggestion is derived from `board.legal_moves` at that exact moment.
- **Multi-cloud AI backend:** two isolated clients — official `google-genai` for Gemini structured JSON, official `openai` client for the Microsoft side (Azure OpenAI or GitHub Models on Azure AI infrastructure), both lazily initialized and running in worker threads so the game loop never blocks.
- **Stale-UI-proof:** control messages are deleted and recreated at every phase change so leftover Discord reactions can't desync the game.

Built solo in one weekend for a hackathon (Microsoft track + MLH Best Use of Gemini API).
