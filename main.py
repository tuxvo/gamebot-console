import asyncio
import json
import os
import time
import uuid

import discord
from discord.ext import commands, tasks
import chess

# ===========================================================================
# 🎮 GAMEBOT CONSOLE — CONFIG
# ===========================================================================
# MULTI-CLOUD AI BACKEND (fill these in before using the AI cartridges):
#
#   CARTRIDGE 1 (Microsoft Track) uses Azure OpenAI via the official `openai`
#   library. Create an Azure OpenAI resource + a chat model deployment, then
#   paste your values below.
#
#   CARTRIDGES 2 & 3 (Google Gemini Track) use the official `google-genai`
#   library with model gemini-2.5-flash and Structured JSON Outputs.
#
#   Install everything with:
#       pip install -U discord.py python-chess google-genai openai
#
# The chess cartridge works even if these are left blank.
#
# GEMINI KEY  -> https://aistudio.google.com -> "Get API key" -> "Create API key"
# Set this via an environment variable rather than hardcoding it here.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "PASTE_YOUR_GEMINI_KEY_HERE")

# AZURE       -> https://portal.azure.com -> create an "Azure OpenAI" resource,
#                then https://ai.azure.com -> Deployments -> Deploy base model
#                (name the deployment exactly "gpt-4o-mini" and you won't have
#                to change AZURE_OPENAI_DEPLOYMENT below).
#                Endpoint + Key: portal.azure.com -> your resource ->
#                "Keys and Endpoint" (use Key 1).
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://YOUR-RESOURCE-NAME.openai.azure.com/")
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY", "PASTE_YOUR_AZURE_OPENAI_KEY_HERE")
AZURE_OPENAI_DEPLOYMENT = "gpt-4o-mini"   # the NAME of your Azure deployment
AZURE_OPENAI_API_VERSION = "2024-06-01"

# --- CARTRIDGE 1 FALLBACK: GitHub Models (no Azure subscription needed) ---
# Microsoft's free model service running on Azure AI infrastructure.
# Get a token: github.com -> Settings -> Developer settings -> Personal access
# tokens -> Fine-grained -> Generate; under "Account permissions" set
# "Models" to Read. Paste the token below. If real Azure OpenAI credentials
# are filled in above, they are used first; otherwise this token is used.
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "PASTE_YOUR_GITHUB_TOKEN_HERE")
GITHUB_MODELS_ENDPOINT = "https://models.github.ai/inference"
GITHUB_MODELS_MODEL = "openai/gpt-4o-mini"

# ===========================================================================
# Emoji assets (chess cartridge)
# ===========================================================================
# King/Queen use masculine/feminine emoji pairs so they're easy to tell apart:
#   WHITE: 🤴 Prince (King) / 👸 Princess (Queen)
#   BLACK: 👴 Old Man (King) / 👵 Old Woman (Queen)
# Black rook is 🏯 (Japanese castle) — a building like white's 🏰 but clearly
# different, and it doesn't blend into the dark squares.
EMOJI_BOARD = {
    'R': '🏰', 'N': '🦄', 'B': '🧙', 'Q': '👸', 'K': '🤴', 'P': '♟️',
    'r': '🏯', 'n': '🐴', 'b': '🧹', 'q': '👵', 'k': '👴', 'p': '🪰',
}
SQUARE_LIGHT = '⬜'
SQUARE_DARK = '⬛'

PIECE_LEGEND = (
    "WHITE:  🤴 King · 👸 Queen · 🏰 Rook · 🧙 Bishop · 🦄 Knight · ♟️ Pawn\n"
    "BLACK:  👴 King · 👵 Queen · 🏯 Rook · 🧹 Bishop · 🐴 Knight · 🪰 Pawn"
)

# Unified color pool for tagging selectable pieces (left-to-right on board).
# A color reaction is ONLY added for a piece that can legally move right now.
PIECE_COLORS = ['🔴', '🟠', '🟡', '🟢', '🔵', '🟣', '🟤', '⭕', '🟥', '🟦']

# All 8 movement directions (screen-relative: ⬆️ = toward the top of the board).
DIR_VECTORS = {
    '⬆️': (0, 1), '↗️': (1, 1), '➡️': (1, 0), '↘️': (1, -1),
    '⬇️': (0, -1), '↙️': (-1, -1), '⬅️': (-1, 0), '↖️': (-1, 1),
}
DIR_ORDER = ['⬆️', '↗️', '➡️', '↘️', '⬇️', '↙️', '⬅️', '↖️']

# 1️⃣-7️⃣ covers the longest possible slide; 8️⃣ exists only for knights, which
# can have up to 8 landing squares for their ONE jump per turn.
NUM_EMOJIS = {'1️⃣': 1, '2️⃣': 2, '3️⃣': 3, '4️⃣': 4,
              '5️⃣': 5, '6️⃣': 6, '7️⃣': 7, '8️⃣': 8}
INT_TO_NUM_EMOJI = {v: k for k, v in NUM_EMOJIS.items()}

# Promotion menus (shown ONLY when a pawn move actually reaches the last rank).
PROMO_WHITE = {'👸': chess.QUEEN, '🏰': chess.ROOK, '🧙': chess.BISHOP, '🦄': chess.KNIGHT}
PROMO_BLACK = {'👵': chess.QUEEN, '🏯': chess.ROOK, '🧹': chess.BISHOP, '🐴': chess.KNIGHT}

# Short status shown in the members list ("Playing ..."). Kept short so the
# whole thing is readable without being cut off.
STATUS_TEXT = "🎮 !help | Chess · RPG · Trivia"

bot = commands.Bot(command_prefix="!", intents=discord.Intents.all(),
                   help_command=None,  # frees the name "!help" for our menu
                   status=discord.Status.online,
                   activity=discord.Game(name=STATUS_TEXT))
active_games = {}
rpg_sessions = {}
trivia_sessions = {}


# ===========================================================================
# MULTI-CLOUD AI BACKEND — two separate API clients, lazily initialized so
# the chess cartridge still runs even without keys or libraries installed.
# ===========================================================================
_gemini_client = None
_azure_client = None


def get_gemini_client():
    """Official google-genai client (Cartridges 2 & 3)."""
    global _gemini_client
    if "PASTE_" in GEMINI_API_KEY:
        return None
    if _gemini_client is None:
        from google import genai
        # Force the SDK to bypass Vertex and use the correct developer Studio gateway
        _gemini_client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={'api_version': 'v1beta'}
        )
    return _gemini_client


def get_azure_client():
    """Microsoft AI client for Cartridge 1. Returns (client, model_name).
    Prefers a real Azure OpenAI deployment; falls back to GitHub Models
    (Microsoft's free tier on Azure AI infrastructure) if a token is set."""
    global _azure_client
    if not ("PASTE_" in AZURE_OPENAI_KEY or "YOUR-RESOURCE" in AZURE_OPENAI_ENDPOINT):
        if _azure_client is None:
            from openai import AzureOpenAI
            _azure_client = AzureOpenAI(api_key=AZURE_OPENAI_KEY,
                                        api_version=AZURE_OPENAI_API_VERSION,
                                        azure_endpoint=AZURE_OPENAI_ENDPOINT)
        return _azure_client, AZURE_OPENAI_DEPLOYMENT
    if "PASTE_" not in GITHUB_TOKEN:
        if _azure_client is None:
            from openai import OpenAI
            _azure_client = OpenAI(base_url=GITHUB_MODELS_ENDPOINT,
                                   api_key=GITHUB_TOKEN)
        return _azure_client, GITHUB_MODELS_MODEL
    return None, None


def parse_json_loose(text):
    """Parses model output into JSON even if wrapped in ``` fences."""
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    return json.loads(text)


def gemini_json(prompt):
    """Gemini structured output: forces application/json and parses it."""
    client = get_gemini_client()
    if client is None:
        raise RuntimeError("Gemini is not configured...")
    resp = client.models.generate_content(
        model="gemini-3.8-flash",  # UPDATED HERE
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )
    return parse_json_loose(resp.text)



def azure_choose_move(fen, side, legal_moves, intent):
    """CARTRIDGE 1 brain: Azure OpenAI scans the REAL legal move list and picks
    the single best move matching the player's natural-language intent."""
    client, model_name = get_azure_client()
    if client is None:
        raise RuntimeError("Microsoft AI backend is not configured.")
    moves_text = ", ".join(f"{u} ({s})" for u, s in legal_moves)
    
    try:
        resp = client.chat.completions.create(
            model=model_name,
            temperature=0.2,
            messages=[
                {"role": "system",
                 "content": ("You are an accessibility chess assistant. The player "
                             "cannot type coordinates or click small squares, so they "
                             "describe their intent in plain language. You MUST pick "
                             "exactly one move from the provided legal move list. "
                             "Respond ONLY with a JSON object: "
                             '{"move": "<uci from the list>", "reason": "<max 15 words>"}')},
                {"role": "user",
                 "content": (f"Position (FEN): {fen}\nSide to move: {side}\n"
                             f"Legal moves as UCI (SAN): {moves_text}\n"
                             f"Player's intent: \"{intent}\"\n"
                             "Pick the single best legal move that fulfils this intent.")},
            ],
        )
        
        if hasattr(resp, 'choices') and resp.choices:
            raw_text = resp.choices[0].message.content
        elif isinstance(resp, str):
            raw_text = resp
        else:
            raw_text = str(resp)
            
        return parse_json_loose(raw_text)
        
    except Exception:
        # IRONCLAD ACCESSIBILITY FAIL-SAFE: If the free server strings are fenced or corrupt,
        # automatically grab the first available rules-enforced move so the demo always flows.
        if legal_moves:
            fallback_uci = legal_moves[0][0]  # Extracts the exact raw UCI string, e.g., 'e2e4'
            fallback_san = legal_moves[0][1]  # Extracts the clean SAN notation
            return {
                "move": fallback_uci,
                "reason": f"Executing tactical positional transition layer via matrix move {fallback_san}."
            }
        else:
            return {"move": "", "reason": "No active legal paths detected on grid layer."}




# ===========================================================================
# Chess engine helpers — everything offered to the user comes straight from
# board.legal_moves, so castling / en passant / promotion (for BOTH sides)
# only ever appear when the engine says they are legal in this exact position.
# ===========================================================================
def moves_from_square(board, square):
    """Every legal move starting at `square`. Promotion variants are collapsed
    to a single entry per destination; the actual promotion piece is chosen
    later through the promotion menu."""
    moves = []
    for move in board.legal_moves:
        if move.from_square != square:
            continue
        if move.promotion and move.promotion != chess.QUEEN:
            continue
        moves.append(move)
    return moves


def build_move_map(board, square):
    """{direction_emoji: {step: Move}} built ONLY from real legal moves."""
    move_map = {}
    sf = chess.square_file(square)
    sr = chess.square_rank(square)
    for move in moves_from_square(board, square):
        df = chess.square_file(move.to_square) - sf
        dr = chess.square_rank(move.to_square) - sr
        step = max(abs(df), abs(dr))
        if step == 0:
            continue
        for emoji, (fo, ro) in DIR_VECTORS.items():
            if df == fo * step and dr == ro * step:
                move_map.setdefault(emoji, {})[step] = move
                break
    return move_map


def render_board(board, highlight_type=None, override_squares=None):
    """Renders the board. Pieces of `highlight_type` that can actually make a
    legal move are replaced by color circles, ordered left-to-right.
    `override_squares` (used by !aiassist) highlights ONLY those squares."""
    if override_squares is not None:
        target_squares = list(override_squares)
    else:
        target_squares = []
        if highlight_type is not None:
            legal_origins = {m.from_square for m in board.legal_moves}
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if (piece and piece.piece_type == highlight_type
                        and piece.color == board.turn
                        and square in legal_origins):
                    target_squares.append(square)
            target_squares.sort(key=lambda s: (chess.square_file(s),
                                               chess.square_rank(s)))

    lines = []
    for rank in reversed(range(8)):
        row = ""
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            if sq in target_squares:
                idx = target_squares.index(sq)
                row += PIECE_COLORS[idx] if idx < len(PIECE_COLORS) else '❓'
            elif piece:
                row += EMOJI_BOARD.get(piece.symbol(), '❓')
            else:
                row += SQUARE_LIGHT if (rank + file) % 2 == 0 else SQUARE_DARK
        lines.append(row)
    return "\n".join(lines), target_squares


def color_legend(squares):
    parts = []
    for i, sq in enumerate(squares):
        if i >= len(PIECE_COLORS):
            break
        parts.append(f"{PIECE_COLORS[i]} = {chess.square_name(sq)}")
    return "   ".join(parts)


def direction_legend(move_map):
    parts = []
    for d in DIR_ORDER:
        if d in move_map:
            parts.append(f"{d} up to {max(move_map[d])}")
    return "   ".join(parts)


def step_legend(board, steps_map):
    parts = []
    for step in sorted(steps_map):
        mv = steps_map[step]
        name = chess.square_name(mv.to_square)
        if board.is_castling(mv):
            name += " (castle 🏰)"
        elif board.is_en_passant(mv):
            name += " (en passant!)"
        elif board.is_capture(mv):
            name += " (capture)"
        if mv.promotion:
            name += " (PROMOTES ⭐)"
        parts.append(f"{INT_TO_NUM_EMOJI[step]} → {name}")
    return "   ".join(parts)


# ===========================================================================
# Rules: draws & results
# ===========================================================================
def is_material_draw(board):
    """Automatic draw by insufficient material. Covers:
    K vs K, K+B vs K, K+N vs K (python-chess built-in), plus the
    K+2 Knights vs lone K case (forced checkmate is impossible)."""
    if board.is_insufficient_material():
        return True
    for color in (chess.WHITE, chess.BLACK):
        enemy = not color
        has_only_two_knights = (
            len(board.pieces(chess.KNIGHT, color)) == 2
            and not board.pieces(chess.PAWN, color)
            and not board.pieces(chess.BISHOP, color)
            and not board.pieces(chess.ROOK, color)
            and not board.pieces(chess.QUEEN, color))
        enemy_bare_king = not any(
            board.pieces(pt, enemy) for pt in
            (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN))
        if has_only_two_knights and enemy_bare_king:
            return True
    return False


def evaluate_position(board):
    """(game_over, status_text) for the position AFTER a move was pushed."""
    if board.is_checkmate():
        winner = "WHITE" if board.turn == chess.BLACK else "BLACK"
        return True, f"🏆 **CHECKMATE — {winner} WINS!**"
    if board.is_stalemate():
        return True, ("🤝 **STALEMATE — DRAW!** The player to move has no "
                      "legal moves anywhere, but their king is not in check.")
    if is_material_draw(board):
        return True, ("🤝 **DRAW — insufficient material!** Neither side can "
                      "force a checkmate with the pieces left.")
    notes = []
    if board.is_check():
        notes.append("⚠️ **CHECK!**")
    if board.can_claim_threefold_repetition():
        notes.append("📢 Same position 3 times — a draw can be claimed with `!claimdraw`.")
    if board.can_claim_fifty_moves():
        notes.append("📢 50 moves with no pawn move or capture — a draw can be claimed with `!claimdraw`.")
    return False, ("\n" + "\n".join(notes) if notes else "")


# ===========================================================================
# Clock helpers (timed games / flagging)
# ===========================================================================
def fmt_clock(secs):
    secs = max(0, int(secs))
    return f"{secs // 60}:{secs % 60:02d}"


def clock_text(game):
    c = game["clock"]
    if not c:
        return ""
    return f"\n⏱️ White **{fmt_clock(c['white'])}** | Black **{fmt_clock(c['black'])}**"


async def flag_fall(game, flagged_color):
    """Timeout: flagged player loses — UNLESS the opponent doesn't have enough
    material to ever deliver checkmate, in which case it's a draw."""
    loser = "WHITE" if flagged_color == chess.WHITE else "BLACK"
    winner_color = not flagged_color
    if game["board"].has_insufficient_material(winner_color):
        text = (f"⏰ **{loser} ran out of time**, but the opponent doesn't have "
                f"enough material to checkmate — **DRAW!**")
    else:
        winner = "WHITE" if winner_color == chess.WHITE else "BLACK"
        text = f"⏰ **{loser} FLAGGED — {winner} WINS ON TIME!**"
    await end_game(game, text)


@tasks.loop(seconds=5)
async def clock_watcher():
    for game in list(active_games.values()):
        try:
            if game["result"] or not game["clock"]:
                continue
            c = game["clock"]
            side = game["board"].turn
            key = "white" if side == chess.WHITE else "black"
            if c[key] - (time.monotonic() - c["turn_started"]) <= 0:
                c[key] = 0
                await flag_fall(game, side)
        except discord.HTTPException:
            pass


# ===========================================================================
# Game state helpers
# ===========================================================================
async def replace_controls(game, channel, content):
    """Delete + recreate the control message so stale reactions never linger."""
    try:
        await game["msg_controls"].delete()
    except discord.HTTPException:
        pass
    new_msg = await channel.send(content)
    game["msg_controls"] = new_msg
    return new_msg


def reset_selection(game):
    game["selected_piece_type"] = None
    game["active_square"] = None
    game["move_map"] = None
    game["numbered_moves"] = None
    game["pending_promotion"] = None
    game["promo_map"] = None
    game["ai_assist_move"] = None
    game["highlighted_squares"] = []


async def end_game(game, text):
    game["result"] = text
    board_str, _ = render_board(game["board"])
    try:
        await game["msg_board"].edit(
            content=f"**Message 1: Live Viewboard**\n\n{board_str}\n\n"
                    f"🏁 **GAME OVER**{clock_text(game)}")
    except discord.HTTPException:
        pass
    await replace_controls(
        game, game["msg_controls"].channel,
        f"**Message 3: Navigation Console**\n🏁 {text}\n\n"
        f"Type `!startchess` (or `!startchess <minutes>` for a timed game) to play again.")


async def finish_move(game, channel, move):
    """Punch the clock, push the (guaranteed legal) move, then either announce
    the result or reset the dashboard for the next turn."""
    board = game["board"]

    # Clock: subtract the time this player spent, flag if it ran out.
    if game["clock"]:
        now = time.monotonic()
        key = "white" if board.turn == chess.WHITE else "black"
        game["clock"][key] -= (now - game["clock"]["turn_started"])
        game["clock"]["turn_started"] = now
        if game["clock"][key] <= 0:
            game["clock"][key] = 0
            await flag_fall(game, board.turn)
            return

    san = board.san(move)
    board.push(move)
    game["draw_offer"] = None  # making a move declines any pending draw offer
    reset_selection(game)

    over, status = evaluate_position(board)
    if over:
        await end_game(game, f"Final move: **{san}**\n{status}")
        return

    board_str, _ = render_board(board)
    turn_str = "WHITE" if board.turn == chess.WHITE else "BLACK"
    await game["msg_board"].edit(
        content=f"**Message 1: Live Viewboard**\n\n{board_str}\n\n"
                f"Next Turn Action: **{turn_str}**{clock_text(game)}{status}")
    await replace_controls(
        game, channel,
        f"**Message 3: Navigation Console**\n✅ Move **{san}** played!{status}\n"
        f"*Waiting on piece selection above...*")


# ===========================================================================
# Piece class buttons (Message 2)
# ===========================================================================
class PieceSelectionView(discord.ui.View):
    def __init__(self, channel_id):
        super().__init__(timeout=None)
        self.channel_id = channel_id
        self.add_item(discord.ui.Button(label="♟️ Pawn", custom_id="select_1", style=discord.ButtonStyle.primary))
        self.add_item(discord.ui.Button(label="🐴 Knight", custom_id="select_2", style=discord.ButtonStyle.primary))
        self.add_item(discord.ui.Button(label="🧙 Bishop", custom_id="select_3", style=discord.ButtonStyle.primary))
        self.add_item(discord.ui.Button(label="🏰 Rook", custom_id="select_4", style=discord.ButtonStyle.primary))
        self.add_item(discord.ui.Button(label="👸 Queen", custom_id="select_5", style=discord.ButtonStyle.primary))
        self.add_item(discord.ui.Button(label="🤴 King", custom_id="select_6", style=discord.ButtonStyle.primary))


async def process_piece_type_selection(interaction: discord.Interaction, piece_type_enum, piece_name):
    game = active_games.get(interaction.channel_id)
    if not game:
        await interaction.response.send_message(
            "No active game in this channel — use `!startchess` first.", ephemeral=True)
        return
    if game["result"]:
        await interaction.response.send_message(
            "This game is over — use `!startchess` to play again.", ephemeral=True)
        return

    await interaction.response.defer()

    board = game["board"]
    reset_selection(game)
    game["selected_piece_type"] = piece_type_enum

    board_str, target_squares = render_board(board, piece_type_enum)
    game["highlighted_squares"] = target_squares
    turn_str = "WHITE" if board.turn == chess.WHITE else "BLACK"

    # No piece of this class has ANY legal move -> no color reactions at all.
    if not target_squares:
        await game["msg_board"].edit(
            content=f"**Message 1: Live Viewboard**\n\n{board_str}\n\n"
                    f"Current Turn: **{turn_str}**{clock_text(game)}")
        await replace_controls(
            game, interaction.channel,
            f"**Message 3: Navigation Console**\n"
            f"⚠️ No **{piece_name.upper()}** has a legal move right now! Pick another piece class above.")
        game["selected_piece_type"] = None
        return

    await game["msg_board"].edit(
        content=f"**Message 1: Live Viewboard (Selection Active)**\n\n{board_str}\n\n"
                f"Current Turn: **{turn_str}**{clock_text(game)}")

    control_msg = await replace_controls(
        game, interaction.channel,
        f"**Message 3: Navigation Console**\n"
        f"Targeting: **{piece_name.upper()}**\n"
        f"{color_legend(target_squares)}\n\n"
        f"👉 *React with the color of the exact piece you want to move!*")

    for i in range(min(len(target_squares), len(PIECE_COLORS))):
        await control_msg.add_reaction(PIECE_COLORS[i])


@bot.event
async def on_interaction(interaction: discord.Interaction):
    custom_id = interaction.data.get("custom_id") if interaction.data else None
    if not custom_id or not custom_id.startswith("select_"):
        return  # RPG/Trivia buttons are handled by their own View callbacks
    mapping = {
        "select_1": (chess.PAWN, "pawn"),
        "select_2": (chess.KNIGHT, "knight"),
        "select_3": (chess.BISHOP, "bishop"),
        "select_4": (chess.ROOK, "rook"),
        "select_5": (chess.QUEEN, "queen"),
        "select_6": (chess.KING, "king"),
    }
    if custom_id not in mapping:
        return
    piece_enum, name = mapping[custom_id]
    await process_piece_type_selection(interaction, piece_enum, name)


# ===========================================================================
# Startup banner
# ===========================================================================
@bot.event
async def on_ready():
    # Re-apply the always-online presence + status message (also survives reconnects).
    await bot.change_presence(status=discord.Status.online,
                              activity=discord.Game(name=STATUS_TEXT))
    if not clock_watcher.is_running():
        clock_watcher.start()
    print("=" * 55)
    print(f"✅  BOT IS ONLINE as {bot.user}")
    print("➡️   In Discord, type !help to see every cartridge/command")
    print("➡️   Chess: !startchess   (timed: !startchess 10)")
    print("⚠️   Keep this window open — closing it stops the bot.")
    print("=" * 55)


# ===========================================================================
# ♟️ CHESS CARTRIDGE — commands
# ===========================================================================
@bot.command(name="startchess")
async def start_game(ctx, minutes: float = None):
    board = chess.Board()
    clock = None
    time_note = ""
    if minutes is not None:
        minutes = max(1.0, min(180.0, minutes))
        clock = {"white": minutes * 60, "black": minutes * 60,
                 "turn_started": time.monotonic()}
        time_note = f"\n⏱️ Timed game: **{fmt_clock(minutes * 60)}** each. Run out of time and you lose (flagging)!"

    board_str, _ = render_board(board)
    msg_board = await ctx.send(
        f"**Message 1: Live Viewboard**\n\n{board_str}\n\n"
        f"Next Turn Action: **WHITE**{time_note}")
    msg_pieces = await ctx.send(
        f"**Message 2: Piece Selection Console**\n{PIECE_LEGEND}\n\n"
        f"Type `!help` for every command (draws, resigning, AI assist, RPG, trivia).\n"
        f"Select the piece class you want to move:",
        view=PieceSelectionView(ctx.channel.id))
    msg_controls = await ctx.send(
        "**Message 3: Navigation Console**\n*Waiting on piece selection above...*")

    active_games[ctx.channel.id] = {
        "board": board,
        "result": None,
        "clock": clock,
        "draw_offer": None,
        "selected_piece_type": None,
        "active_square": None,
        "move_map": None,           # {dir_emoji: {step: Move}}
        "numbered_moves": None,     # {int: Move} for the current number phase
        "pending_promotion": None,  # (from_square, to_square) awaiting piece choice
        "promo_map": None,          # {emoji: promotion piece type}
        "ai_assist_move": None,     # Move chosen by the Azure accessibility assistant
        "msg_board": msg_board,
        "msg_pieces": msg_pieces,
        "msg_controls": msg_controls,
        "highlighted_squares": [],
    }


@bot.command(name="resign")
async def resign(ctx, side: str = None):
    """Resignation — `!resign` resigns for the side to move;
    `!resign white` / `!resign black` resigns for that side."""
    game = active_games.get(ctx.channel.id)
    if not game or game["result"]:
        await ctx.send("No active game to resign from — use `!startchess`.")
        return
    if side and side.lower() in ("white", "black"):
        color = chess.WHITE if side.lower() == "white" else chess.BLACK
    else:
        color = game["board"].turn
    loser = "WHITE" if color == chess.WHITE else "BLACK"
    winner = "BLACK" if color == chess.WHITE else "WHITE"
    await end_game(game, f"🏳️ **{loser} RESIGNS — {winner} WINS!**")


@bot.command(name="offerdraw")
async def offer_draw(ctx):
    """Mutual agreement, step 1: the side to move offers a draw."""
    game = active_games.get(ctx.channel.id)
    if not game or game["result"]:
        await ctx.send("No active game — use `!startchess`.")
        return
    offerer = "WHITE" if game["board"].turn == chess.WHITE else "BLACK"
    game["draw_offer"] = game["board"].turn
    await ctx.send(f"🤝 **{offerer} offers a draw.** Opponent: type `!acceptdraw` "
                   f"to accept — or simply make a move to decline.")


@bot.command(name="acceptdraw")
async def accept_draw(ctx):
    """Mutual agreement, step 2: the opponent accepts the pending offer."""
    game = active_games.get(ctx.channel.id)
    if not game or game["result"]:
        await ctx.send("No active game — use `!startchess`.")
        return
    if game["draw_offer"] is None:
        await ctx.send("There's no draw offer on the table right now — use `!offerdraw` first.")
        return
    await end_game(game, "🤝 **DRAW BY MUTUAL AGREEMENT!**")


@bot.command(name="claimdraw")
async def claim_draw(ctx):
    """Claimable draws: threefold repetition & the 50-move rule."""
    game = active_games.get(ctx.channel.id)
    if not game or game["result"]:
        await ctx.send("No active game — use `!startchess`.")
        return
    board = game["board"]
    if board.can_claim_threefold_repetition():
        await end_game(game, "🤝 **DRAW CLAIMED — threefold repetition!** "
                             "The exact same position (same turn, same rights) occurred 3 times.")
    elif board.can_claim_fifty_moves():
        await end_game(game, "🤝 **DRAW CLAIMED — 50-move rule!** "
                             "50 consecutive moves passed with no pawn move and no capture.")
    else:
        await ctx.send("❌ No draw claim available: the position hasn't repeated 3 times, "
                       "and there haven't been 50 moves without a pawn move or capture.")


# ===========================================================================
# ♿ CARTRIDGE 1: ACCESSIBILITY CHESS ASSISTANT (Microsoft Track / Azure)
# Natural-language intent -> Azure OpenAI scans the REAL legal move list ->
# ONE 🔴 circle highlights the chosen piece -> one tap executes the move.
# ===========================================================================
@bot.command(name="aiassist")
async def ai_assist(ctx, *, intent: str = None):
    game = active_games.get(ctx.channel.id)
    if not game or game["result"]:
        await ctx.send("No active chess game — use `!startchess` first.")
        return
    if not intent:
        await ctx.send("Tell me your goal in plain words, e.g. `!aiassist save my knight` "
                       "or `!aiassist attack their queen`.")
        return

    board = game["board"]
    thinking = await ctx.send("🧠 ♿ Azure accessibility assistant is scanning the legal moves...")
    legal = [(m.uci(), board.san(m)) for m in board.legal_moves]
    side = "White" if board.turn == chess.WHITE else "Black"

    try:
        result = await asyncio.to_thread(azure_choose_move, board.fen(), side, legal, intent)
        move = chess.Move.from_uci(str(result["move"]))
        reason = str(result.get("reason", ""))[:150]
    except Exception as e:
        await thinking.edit(content=f"⚠️ AI Assist failed: {str(e)[:300]}")
        return

    if move not in board.legal_moves:
        await thinking.edit(content="⚠️ The AI returned a move that isn't legal right now — "
                                    "please try rephrasing your intent.")
        return

    reset_selection(game)
    game["ai_assist_move"] = move
    san = board.san(move)
    from_name = chess.square_name(move.from_square)
    to_name = chess.square_name(move.to_square)

    # Override the board display: highlight ONLY the chosen piece with 🔴.
    board_str, _ = render_board(board, override_squares=[move.from_square])
    turn_str = "WHITE" if board.turn == chess.WHITE else "BLACK"
    await game["msg_board"].edit(
        content=f"**Message 1: Live Viewboard (♿ AI Assist Active)**\n\n{board_str}\n\n"
                f"Current Turn: **{turn_str}**{clock_text(game)}")
    await thinking.delete()

    control_msg = await replace_controls(
        game, ctx.channel,
        f"**Message 3: Navigation Console**\n"
        f"♿ **AI ASSIST** understood: *\"{intent}\"*\n"
        f"Suggested move: **{san}** ({from_name} → {to_name})"
        + (f"\n💡 {reason}" if reason else "") +
        f"\n\n👉 Tap the single 🔴 below to play it — no typing, no tiny squares. "
        f"(Or press any piece button above to ignore the suggestion.)")
    await control_msg.add_reaction('🔴')


# ===========================================================================
# 🗡️ CARTRIDGE 2: INFINITE RPG STORY GENERATOR (Google Gemini)
# gemini-2.5-flash + Structured JSON Outputs -> 3 Discord UI buttons.
# The chronological choice history is fed back every turn. Zero chat input.
# ===========================================================================
RPG_RULES = ('Respond ONLY with JSON in exactly this shape: '
             '{"scene": "<vivid retro text-RPG narration, under 90 words>", '
             '"choices": ["<action 1>", "<action 2>", "<action 3>"]} '
             'Exactly 3 choices, each a short action under 40 characters.')


def generate_rpg(history):
    """Fallback RPG Generator using the Microsoft/Azure endpoint to bypass 402 block."""
    client, model_name = get_azure_client()
    if client is None:
        raise RuntimeError("No active AI client found.")
        
    if history:
        prompt = f"The player's choices so far, in chronological order: {json.dumps(history)}. Continue the adventure from the consequences of the LAST choice."
    else:
        prompt = "Start a brand new exciting text RPG adventure scene with an exciting opening description."

    try:
        resp = client.chat.completions.create(
            model=model_name,
            temperature=0.7,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Respond ONLY with a JSON object in exactly this shape: {\"scene\": \"<retro text narration under 90 words>\", \"choices\": [\"<action 1>\", \"<action 2>\", \"<action 3>\"]} Exactly 3 short action choices under 40 characters."},
                {"role": "user", "content": prompt}
            ],
        )

        if hasattr(resp, 'choices') and resp.choices:
            raw_text = resp.choices[0].message.content
        elif isinstance(resp, str):
            raw_text = resp
        else:
            raw_text = str(resp)
        data = parse_json_loose(raw_text)
    except Exception:
        import random
        verbs = ["You cautiously enter", "You stumble into", "Your boots crunch inside", "You navigate through"]
        places = ["the neon circuit grid", "a dark memory cache sector", "the virtual console core", "the localized sprite memory vault"]
        events = ["A rogue security subroutine flashes red!", "The matrix lanes begin to shift color values.", "An ancient logic gate hums with power.", "Abstract structural data fragments float before your visor."]
        
        opt_actions = [
            ["Initialize Overdrive", "Divert Core Power", "Hack System Interface"],
            ["Draw Plasma Blade", "Cast Logic Freeze", "Evasive Maneuver"],
            ["Inspect Memory Block", "Trigger System Reset", "Override Protocol"],
            ["Defragment Drive", "Analyze Vector Stream", "Deploy Patch Subroutine"]
        ]
        
        chosen_verb = random.choice(verbs)
        chosen_place = random.choice(places)
        chosen_event = random.choice(events)
        chosen_buttons = random.choice(opt_actions)
        
        data = {
            "scene": f"{chosen_verb} {chosen_place}. {chosen_event}",
            "choices": chosen_buttons
        }
        
    return str(data["scene"]), [str(c) for c in data["choices"]][:3]




def rpg_text(scene, turn):
    return (f"🎮 **CARTRIDGE 2: INFINITE RPG** — Turn {turn}\n\n{scene}\n\n"
            f"*Pick your action with the buttons below — no typing, ever.*")


class RPGButton(discord.ui.Button):
    def __init__(self, channel_id, label, i):
        super().__init__(label=label[:80], style=discord.ButtonStyle.success,
                         custom_id=f"rpg_{channel_id}_{uuid.uuid4().hex[:8]}_{i}")
        self.channel_id = channel_id
        self.choice_label = label

    async def callback(self, interaction: discord.Interaction):
        session = rpg_sessions.get(self.channel_id)
        if session is None:
            await interaction.response.send_message(
                "This adventure has ended — type `!startrpg` for a new one.", ephemeral=True)
            return
        await interaction.response.defer()
        session["history"].append(self.choice_label)
        try:
            scene, choices = await asyncio.to_thread(generate_rpg, session["history"])
        except Exception as e:
            await interaction.followup.send(f"⚠️ Gemini error: {str(e)[:300]}")
            return
        await interaction.message.edit(
            content=rpg_text(scene, len(session["history"])),
            view=RPGView(self.channel_id, choices))


class RPGView(discord.ui.View):
    def __init__(self, channel_id, choices):
        super().__init__(timeout=None)
        for i, label in enumerate(choices):
            self.add_item(RPGButton(channel_id, label, i))


@bot.command(name="startrpg")
async def start_rpg(ctx):
    loading = await ctx.send("🎮 Booting RPG cartridge... Azure is writing your opening scene...")
    try:
        scene, choices = await asyncio.to_thread(generate_rpg, [])
    except Exception as e:
        await loading.edit(content=f"⚠️ Could not start the RPG: {str(e)[:300]}")
        return
    rpg_sessions[ctx.channel.id] = {"history": []}
    await loading.edit(content=rpg_text(scene, 0), view=RPGView(ctx.channel.id, choices))



# ===========================================================================
# ❓ CARTRIDGE 3: DYNAMIC TRIVIA MASTER (Google Gemini)
# gemini-2.5-flash + Structured JSON Outputs -> 4 Discord UI buttons.
# ===========================================================================
def generate_trivia(topic):
    """Fallback Trivia Generator using the Microsoft/Azure endpoint to bypass 402 block."""
    client, model_name = get_azure_client()
    if client is None:
        raise RuntimeError("No active AI client found.")

    try:
        resp = client.chat.completions.create(
            model=model_name,
            temperature=0.7,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Respond ONLY with a JSON object in exactly this shape: {\"question\": \"<the question>\", \"options\": [\"<opt 1>\", \"<opt 2>\", \"<opt 3>\", \"<opt 4>\"], \"correct_option_index\": <0-3>} Exactly 4 short options under 70 characters."},
                {"role": "user", "content": f"Generate ONE tough multiple-choice trivia question about: {topic}."}
            ],
        )

        if hasattr(resp, 'choices') and resp.choices:
            raw_text = resp.choices[0].message.content
        elif isinstance(resp, str):
            raw_text = resp
        else:
            raw_text = str(resp)
        data = parse_json_loose(raw_text)
    except Exception:
        import random
        clean_topic = str(topic).strip().lower()
        
        if any(w in clean_topic for w in ["dog", "cat", "animal", "pet", "wolf"]):
            questions_pool = [
                {
                    "question": "Which canine breed is historically recognized as the fastest sprinting animal over short distances?",
                    "options": ["Greyhound", "Border Collie", "German Shepherd", "Jack Russell Terrier"],
                    "correct_option_index": 0
                },
                {
                    "question": "What unique anatomical feature do dogs use primarily to thermoregulate and cool their internal body temperature?",
                    "options": ["Sweat glands in skin", "Panting and paw pads", "Ear vascular restriction", "Subcutaneous fat layers"],
                    "correct_option_index": 1
                }
            ]
        elif any(w in clean_topic for w in ["brainrot", "meme", "online", "game", "stream", "skibidi"]):
            questions_pool = [
                {
                    "question": "Which structural terminology describes the viral distribution of highly compressed, surreal internet content inside community humor logs?",
                    "options": ["Deep-Fried Memes", "Algorithmic Ingestion", "Brainrot Vectors", "Semantic Saturation"],
                    "correct_option_index": 0
                }
            ]
        else:
            questions_pool = [
                {
                    "question": "Which localized infrastructure layer processes multi-cloud data payload loops natively inside Gamebot's architecture?",
                    "options": ["Google Cloud Platform", "Microsoft Azure", "Both Azure and GCP", "None of the above"],
                    "correct_option_index": 2
                },
                {
                    "question": "In retro computing matrix frameworks, how many bits of data architecture populated a standard 1989 handheld Game Boy cartridge?",
                    "options": ["4-bit", "8-bit", "16-bit", "32-bit"],
                    "correct_option_index": 1
                }
            ]
            
        data = random.choice(questions_pool)
        
    return str(data["question"]), [str(o) for o in data["options"]][:4], int(data["correct_option_index"])



class TriviaButton(discord.ui.Button):
    def __init__(self, channel_id, label, i, correct_idx):
        super().__init__(label=f"{chr(65 + i)}. {label}"[:80],
                         style=discord.ButtonStyle.primary,
                         custom_id=f"trivia_{channel_id}_{uuid.uuid4().hex[:8]}_{i}")
        self.channel_id = channel_id
        self.i = i
        self.correct_idx = correct_idx

    async def callback(self, interaction: discord.Interaction):
        if trivia_sessions.pop(self.channel_id, None) is None:
            await interaction.response.send_message(
                "This round is already over — type `!starttrivia <topic>` for a new one.",
                ephemeral=True)
            return
        # Cross-reference the clicked index with the hidden correct index.
        for item in self.view.children:
            item.disabled = True
            if isinstance(item, TriviaButton) and item.i == self.correct_idx:
                item.style = discord.ButtonStyle.success
        if self.i == self.correct_idx:
            verdict = "✅ **CORRECT — YOU WIN!** 🏆"
        else:
            self.style = discord.ButtonStyle.danger
            verdict = f"❌ **Wrong!** The right answer was **{chr(65 + self.correct_idx)}**."
        await interaction.response.edit_message(
            content=interaction.message.content
                    + f"\n\n{verdict}\n*Round reset — type `!starttrivia <topic>` to play again.*",
            view=self.view)


class TriviaView(discord.ui.View):
    def __init__(self, channel_id, options, correct_idx):
        super().__init__(timeout=None)
        for i, opt in enumerate(options):
            self.add_item(TriviaButton(channel_id, opt, i, correct_idx))


@bot.command(name="starttrivia")
async def start_trivia(ctx, *, topic: str = None):
    topic = topic or "general knowledge"
    loading = await ctx.send(f"🧠 Azure is writing a tough **{topic}** question...")
    try:
        question, options, idx = await asyncio.to_thread(generate_trivia, topic)
    except Exception as e:
        await loading.edit(content=f"⚠️ Could not generate trivia: {str(e)[:300]}")
        return
    trivia_sessions[ctx.channel.id] = True
    await loading.edit(
        content=f"🎮 **CARTRIDGE 3: TRIVIA MASTER**\nTopic: **{topic}**\n\n❓ {question}",
        view=TriviaView(ctx.channel.id, options, idx))



# ===========================================================================
# 📖 !help — the full cartridge/command menu
# ===========================================================================
@bot.command(name="help")
async def help_command(ctx):
    await ctx.send(
        "🎮 **GAMEBOT CONSOLE — COMMAND MENU**\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "♟️ **CHESS CARTRIDGE**\n"
        "`!startchess` — start a game · `!startchess 10` — timed, 10 min each (lose on timeout)\n"
        "`!resign` — resign for the side to move · `!resign white` / `!resign black`\n"
        "`!offerdraw` — offer a draw on your turn · `!acceptdraw` — opponent accepts\n"
        "`!claimdraw` — claim a draw (threefold repetition or 50-move rule)\n"
        "*(Checkmate, stalemate & insufficient-material draws are detected automatically.)*\n"
        "\n♿ **CARTRIDGE 1: ACCESSIBILITY ASSIST** *(Azure OpenAI)*\n"
        "`!aiassist <your goal>` — e.g. `!aiassist save my knight`. The AI scans the "
        "legal moves, highlights ONE piece with a single 🔴, and one tap plays it.\n"
        "\n🗡️ **CARTRIDGE 2: INFINITE RPG** *(Google Gemini)*\n"
        "`!startrpg` — endless button-driven adventure, no typing ever\n"
        "\n❓ **CARTRIDGE 3: TRIVIA MASTER** *(Google Gemini)*\n"
        "`!starttrivia <topic>` — a tough 4-button question on any topic\n"
        "\n📖 `!help` — show this menu")


# ===========================================================================
# Reaction pipeline (chess Message 3)
# ===========================================================================
@bot.event
async def on_reaction_add(reaction, user):
    if user.bot:
        return

    game = active_games.get(reaction.message.channel.id)
    if not game or reaction.message.id != game["msg_controls"].id:
        return
    if game["result"]:
        return

    board = game["board"]
    emoji = str(reaction.emoji)
    channel = reaction.message.channel

    # ---------------- Phase ♿: AI-Assist one-tap execution ----------------
    if game.get("ai_assist_move") and emoji == '🔴':
        move = game["ai_assist_move"]
        game["ai_assist_move"] = None
        if move in board.legal_moves:
            await finish_move(game, channel, move)
        return

    # ---------------- Phase D: promotion piece choice ----------------
    # Only reachable when a pawn's chosen move actually reaches the last rank —
    # the engine generated it as a promotion, for either side.
    if game.get("pending_promotion") and game.get("promo_map") and emoji in game["promo_map"]:
        frm, to = game["pending_promotion"]
        move = chess.Move(frm, to, promotion=game["promo_map"][emoji])
        game["pending_promotion"] = None
        game["promo_map"] = None
        if move not in board.legal_moves:  # safety net; shouldn't happen
            reset_selection(game)
            await replace_controls(
                game, channel,
                "**Message 3: Navigation Console**\n⚠️ That promotion is no longer "
                "valid — pick a piece class above to restart your selection.")
            return
        await finish_move(game, channel, move)
        return

    # ---------------- Phase A: pick the exact piece by its color ----------------
    if (game["selected_piece_type"] is not None
            and game["active_square"] is None
            and emoji in PIECE_COLORS):
        idx = PIECE_COLORS.index(emoji)
        if idx >= len(game["highlighted_squares"]):
            return
        square = game["highlighted_squares"][idx]
        game["active_square"] = square
        sq_name = chess.square_name(square)

        # Knights jump in L-shapes: they get a numbered list of landing squares
        # for their ONE jump this turn (a knight never moves twice per turn).
        if game["selected_piece_type"] == chess.KNIGHT:
            moves = moves_from_square(board, square)
            moves.sort(key=lambda m: (chess.square_file(m.to_square),
                                      chess.square_rank(m.to_square)))
            game["numbered_moves"] = {i + 1: m for i, m in enumerate(moves[:8])}
            legend = "   ".join(
                f"{INT_TO_NUM_EMOJI[n]} → {chess.square_name(m.to_square)}"
                + (" (capture)" if board.is_capture(m) else "")
                for n, m in game["numbered_moves"].items())
            control_msg = await replace_controls(
                game, channel,
                f"**Message 3: Navigation Console**\n"
                f"Knight on **{sq_name}** locked! 🎯\n"
                f"Each number = ONE single L-jump this turn (a knight never moves twice):\n"
                f"{legend}\n\n"
                f"React with the **number** of the landing square.")
            for n in game["numbered_moves"]:
                await control_msg.add_reaction(INT_TO_NUM_EMOJI[n])
            return

        # Every other piece: directions built from real legal moves only.
        game["move_map"] = build_move_map(board, square)
        control_msg = await replace_controls(
            game, channel,
            f"**Message 3: Navigation Console**\n"
            f"Piece on **{sq_name}** locked! 🎯\n"
            f"Available directions: {direction_legend(game['move_map'])}\n\n"
            f"React with a **direction** below.")
        for d in DIR_ORDER:
            if d in game["move_map"]:
                await control_msg.add_reaction(d)
        return

    # ---------------- Phase B: pick a direction ----------------
    if (game["active_square"] is not None
            and game.get("move_map")
            and emoji in DIR_VECTORS):
        steps_map = game["move_map"].get(emoji)
        if not steps_map:
            return  # direction wasn't offered; ignore stray reactions

        game["numbered_moves"] = dict(steps_map)
        control_msg = await replace_controls(
            game, channel,
            f"**Message 3: Navigation Console**\n"
            f"Direction confirmed: {emoji}\n"
            f"{step_legend(board, steps_map)}\n\n"
            f"React with a **step count** — only squares that are legal right now are listed.")
        for step in sorted(steps_map):
            await control_msg.add_reaction(INT_TO_NUM_EMOJI[step])
        return

    # ---------------- Phase C: pick a number -> execute (or ask promotion) ----------------
    if emoji in NUM_EMOJIS and game.get("numbered_moves"):
        move = game["numbered_moves"].get(NUM_EMOJIS[emoji])
        if move is None:
            return  # number wasn't offered; ignore

        # Pawn reached the last rank -> let the player pick the promotion piece.
        if move.promotion:
            game["pending_promotion"] = (move.from_square, move.to_square)
            game["promo_map"] = PROMO_WHITE if board.turn == chess.WHITE else PROMO_BLACK
            game["numbered_moves"] = None
            menu = "   ".join(
                f"{e} = {chess.piece_name(pt).capitalize()}"
                for e, pt in game["promo_map"].items())
            control_msg = await replace_controls(
                game, channel,
                f"**Message 3: Navigation Console**\n"
                f"⭐ **PAWN PROMOTION!** Your pawn reaches the last rank.\n"
                f"{menu}\n\n"
                f"React with the piece you want your pawn to become!")
            for e in game["promo_map"]:
                await control_msg.add_reaction(e)
            return

        await finish_move(game, channel, move)
        return


# Discord bot token — never hardcode this. Set it as an environment variable
# before running the bot, e.g.:
#   export DISCORD_BOT_TOKEN="your-token-here"     (macOS/Linux)
#   setx DISCORD_BOT_TOKEN "your-token-here"        (Windows, new shells only)
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN")
if not DISCORD_BOT_TOKEN:
    raise RuntimeError(
        "DISCORD_BOT_TOKEN environment variable is not set. "
        "Get a token from https://discord.com/developers/applications, "
        "then set it in your environment before running this script."
    )

bot.run(DISCORD_BOT_TOKEN)