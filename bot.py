import discord
from discord.ext import commands
import os
import datetime
import re
import asyncio
import json
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from PIL import Image

# --- 設定ファイル名 ---
CONFIG_FILE_NAME = "config.json"

# --- デフォルト設定 ---
DEFAULT_CONFIG = {
    "admin_role_names": ["BOT管理者", "運営スタッフ"],
    "default_gemini_model": "gemini-1.5-flash-latest",
    "tagging_prompt_file": "Tagging_prompt.txt",
    "base_upload_folder": "uploads",
    "max_files_to_send_on_search": 5 # この設定は /files get が削除されるため、直接は使われなくなります
}

# --- 設定読み込み関数 ---
def load_bot_config():
    config = DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_FILE_NAME):
        try:
            with open(CONFIG_FILE_NAME, "r", encoding="utf-8") as f:
                loaded_config = json.load(f)
                config.update(loaded_config)
                print(f"設定ファイルを '{CONFIG_FILE_NAME}' から読み込みました。")
        except json.JSONDecodeError:
            print(f"エラー: '{CONFIG_FILE_NAME}' のJSON形式が正しくありません。デフォルト設定を使用します。")
        except Exception as e:
            print(f"エラー: '{CONFIG_FILE_NAME}' の読み込み中に問題が発生しました: {e}。デフォルト設定を使用します。")
    else:
        print(f"情報: '{CONFIG_FILE_NAME}' が見つかりません。デフォルト設定で作成します。")
        try:
            with open(CONFIG_FILE_NAME, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_CONFIG, f, indent=4, ensure_ascii=False)
            print(f"'{CONFIG_FILE_NAME}' をデフォルト設定で作成しました。")
        except Exception as e:
            print(f"エラー: '{CONFIG_FILE_NAME}' の作成中に問題が発生しました: {e}")
    return config

# --- .envとconfig.jsonから設定を読み込む ---
load_dotenv()
bot_config = load_bot_config()

DISCORD_BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

ADMIN_ROLE_NAMES = bot_config.get("admin_role_names", DEFAULT_CONFIG["admin_role_names"])
DEFAULT_GEMINI_MODEL = bot_config.get("default_gemini_model", DEFAULT_CONFIG["default_gemini_model"])
TAGGING_PROMPT_FILE = bot_config.get("tagging_prompt_file", DEFAULT_CONFIG["tagging_prompt_file"])
BASE_UPLOAD_FOLDER = bot_config.get("base_upload_folder", DEFAULT_CONFIG["base_upload_folder"])
# MAX_FILES_TO_SEND_ON_SEARCH は /files get が削除されるため、直接は使われなくなりますが、設定として残しておきます。

DEFAULT_TAGGING_PROMPT_TEXT = (
    "このファイルの内容を詳細に分析し、関連性の高いキーワードを5つ提案してください。"
    "各キーワードは簡潔な日本語で、ハイフン(-)で連結可能な形式でお願いします。"
    "例: 風景-自然-山-川-晴天"
    "もし内容が不明瞭な場合やキーワード抽出が難しい場合は、「タグ抽出不可」とだけ返してください。"
)

def load_tagging_prompt():
    prompt_file_path = TAGGING_PROMPT_FILE
    if os.path.exists(prompt_file_path):
        try:
            with open(prompt_file_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
                if prompt:
                    print(f"タグ付けプロンプトを '{prompt_file_path}' から読み込みました。")
                    return prompt
                else:
                    print(f"警告: '{prompt_file_path}' は空です。デフォルトのプロンプトを使用します。")
        except Exception as e:
            print(f"警告: '{prompt_file_path}' の読み込みに失敗しました: {e}。デフォルトのプロンプトを使用します。")
    else:
        print(f"情報: '{prompt_file_path}' が見つかりません。デフォルトのプロンプトを使用します。")
    return DEFAULT_TAGGING_PROMPT_TEXT

gemini_model_instance = None
current_gemini_model = DEFAULT_GEMINI_MODEL

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    try:
        gemini_model_instance = genai.GenerativeModel(
            current_gemini_model,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        print(f"Geminiモデル '{current_gemini_model}' の初期化に成功しました。")
    except Exception as e:
        print(f"エラー: デフォルトのGeminiモデル '{current_gemini_model}' の初期化に失敗しました: {e}")
        gemini_model_instance = None
else:
    print("情報: GEMINI_API_KEYが設定されていません。Gemini API関連の機能は利用できません。")

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix='/', intents=intents)

def sanitize_filename_component(text):
    return re.sub(r'[\\/*?:"<>|\s]', '_', text)

# get_file_icon は /files list で使われていたため、現在は直接使われませんが、
# 将来的に何らかの形でファイル情報を示す際に使える可能性があるので残しておきます。
def get_file_icon(extension):
    ext = extension.lower()
    if ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']: return "🖼️"
    elif ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm']: return "🎬"
    elif ext in ['.txt', '.md', '.doc', '.pdf']: return "📄"
    else: return "📁"

def create_year_month_folder_if_not_exists(base_folder_from_config):
    now = datetime.datetime.now()
    year_month_folder_name = now.strftime("%Y%m")
    year_month_folder_path = os.path.join(base_folder_from_config, year_month_folder_name)
    if not os.path.exists(year_month_folder_path):
        os.makedirs(year_month_folder_path)
        print(f"年月フォルダ '{year_month_folder_path}' を作成しました。")
    return year_month_folder_path

def is_admin():
    async def predicate(ctx):
        if ctx.guild is None:
            await ctx.send("このコマンドはサーバー内でのみ実行可能です。")
            return False
        author_roles = [role.name for role in ctx.author.roles]
        if any(admin_role in author_roles for admin_role in ADMIN_ROLE_NAMES):
            return True
        await ctx.send("このコマンドを実行する権限がありません。")
        return False
    return commands.check(predicate)

async def get_tags_from_gemini(file_path, original_filename, mime_type):
    global gemini_model_instance
    if not gemini_model_instance:
        print("Geminiモデルが初期化されていないため、タグ生成をスキップします。")
        return "notags"
    print(f"Gemini APIにファイル '{original_filename}' (MIMEタイプ: {mime_type}) を送信してタグを生成します...")
    try:
        uploaded_file = genai.upload_file(path=file_path, display_name=original_filename)
        prompt = load_tagging_prompt()
        response = await gemini_model_instance.generate_content_async(
            [prompt, uploaded_file],
            generation_config={"response_mime_type": "text/plain"}
        )
        if response.text.strip() == "タグ抽出不可":
            print("Gemini API: タグ抽出不可と判断されました。")
            return "notags"
        tags = response.text.strip()
        sanitized_tags = sanitize_filename_component(tags)
        print(f"Gemini APIから取得したタグ: '{sanitized_tags}'")
        return sanitized_tags if sanitized_tags else "notags"
    except Exception as e:
        print(f"Gemini APIでのタグ生成中にエラーが発生しました: {e}")
        return "notags"
    finally:
        if 'uploaded_file' in locals() and uploaded_file and hasattr(uploaded_file, 'name'):
             try:
                 # genai.delete_file(uploaded_file.name) # SDKのバージョンやポリシーにより検討
                 pass
             except Exception as e_del:
                 print(f"Gemini APIからアップロードされたファイル {uploaded_file.name} の削除中にエラー: {e_del}")

@bot.event
async def on_ready():
    global current_gemini_model
    print(f'{bot.user.name} としてログインしました (ID: {bot.user.id})')
    print(f'監視中のサーバー数: {len(bot.guilds)}')
    print(f'ベースアップロードフォルダ: {os.path.abspath(BASE_UPLOAD_FOLDER)}')
    print(f'管理者ロール: {ADMIN_ROLE_NAMES}')
    if gemini_model_instance:
        print(f'使用中Geminiモデル: {current_gemini_model}')
    else:
        print('Geminiモデルは初期化されていません。')
    load_tagging_prompt()
    if not os.path.exists(BASE_UPLOAD_FOLDER):
        os.makedirs(BASE_UPLOAD_FOLDER)
        print(f"ベースフォルダ '{BASE_UPLOAD_FOLDER}' を作成しました。")
    try:
        await bot.tree.sync()
        print("スラッシュコマンドを同期しました。")
    except Exception as e:
        print(f"スラッシュコマンドの同期に失敗しました: {e}")
    print('------')

@bot.event
async def on_message(message): # ファイルアップロード時の自動処理は残す
    if message.author == bot.user: return
    if message.attachments:
        year_month_folder_path = create_year_month_folder_if_not_exists(BASE_UPLOAD_FOLDER)
        ctx = await bot.get_context(message)
        for attachment in message.attachments:
            allowed_image_types = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp')
            allowed_video_types = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
            file_ext = os.path.splitext(attachment.filename)[1].lower()
            if not (file_ext in allowed_image_types or file_ext in allowed_video_types):
                await message.channel.send(f"ファイル '{attachment.filename}' の形式 ({file_ext}) はサポートされていません。\n対応形式 (画像): {', '.join(allowed_image_types)}\n対応形式 (動画): {', '.join(allowed_video_types)}")
                continue
            if attachment.size > 8 * 1024 * 1024 and not (ctx.guild and ctx.guild.premium_tier >= 1):
                await message.channel.send(f"ファイル '{attachment.filename}' ({attachment.size // 1024 // 1024}MB) はサイズが大きすぎます。サーバーブーストレベルに応じて上限が緩和されますが、基本は8MBまでです。")
                continue
            temp_save_path = os.path.join(year_month_folder_path, f"temp_{attachment.filename}")
            await attachment.save(temp_save_path)
            processing_msg = await message.channel.send(f"ファイル '{attachment.filename}' をアップロード中... 自動タグ付け処理を開始します。しばらくお待ちください。")
            tags_str = "notags"
            if gemini_model_instance:
                try:
                    if file_ext in allowed_image_types:
                        try:
                            img = Image.open(temp_save_path); img.verify(); img.close()
                        except Exception as img_err:
                            await processing_msg.edit(content=f"ファイル '{attachment.filename}' は有効な画像ファイルではないか、破損しているようです。処理を中断します。({img_err})")
                            if os.path.exists(temp_save_path): os.remove(temp_save_path)
                            continue
                    mime_type_for_gemini = attachment.content_type
                    tags_str = await get_tags_from_gemini(temp_save_path, attachment.filename, mime_type_for_gemini)
                except Exception as e:
                    print(f"タグ付け処理中にエラー: {e}")
                    await processing_msg.edit(content=f"ファイル '{attachment.filename}' のタグ付け中にエラーが発生しました。タグなしで保存します。")
                    tags_str = "notags"
            else:
                await processing_msg.edit(content=f"ファイル '{attachment.filename}' をアップロード中... (Gemini APIが未設定のため自動タグ付けはスキップされました)")
            date_str = datetime.datetime.now().strftime("%Y%m%d")
            original_filename_no_ext, original_ext = os.path.splitext(attachment.filename)
            sanitized_original_filename = sanitize_filename_component(original_filename_no_ext)
            new_filename = f"{date_str}_{tags_str}_{sanitized_original_filename}{original_ext}"
            final_save_path = os.path.join(year_month_folder_path, new_filename)
            try:
                os.rename(temp_save_path, final_save_path)
                print(f"ファイル '{attachment.filename}' を '{final_save_path}' に保存しました。")
                await processing_msg.edit(content=(f"ファイル '{attachment.filename}' をアップロードし、'{new_filename}' として保存しました。\n自動タグ: `{tags_str if tags_str != 'notags' else 'なし'}`"))
            except Exception as e:
                print(f"ファイルのリネーム/保存中にエラー: {e}")
                await processing_msg.edit(content=f"ファイル '{attachment.filename}' の最終保存中にエラーが発生しました。")
                if os.path.exists(temp_save_path): os.remove(temp_save_path)
    await bot.process_commands(message)

# --- オートコンプリート用の関数 ---
# year_month_autocomplete と filename_autocomplete は /files グループが削除されたため不要になりました。
# gemini_model_autocomplete は /gemini set で使用するため残します。
async def gemini_model_autocomplete(interaction: discord.Interaction, current: str) -> list[discord.app_commands.Choice[str]]:
    choices = []
    if not GEMINI_API_KEY or not gemini_model_instance: return []
    try:
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                model_display_name = model.name.replace("models/", "")
                if current.lower() in model_display_name.lower():
                    choice_name = f"{model_display_name} ({model.display_name})"
                    if len(choice_name) > 100:
                        choice_name = model_display_name[:97] + "..." if len(model_display_name) > 97 else model_display_name
                    choices.append(discord.app_commands.Choice(name=choice_name, value=model_display_name))
            if len(choices) >= 25: break
    except Exception as e: print(f"Geminiモデルのオートコンプリート中にエラー: {e}")
    return choices

# --- コマンドグループの定義 ---
# files_group は削除
gemini_group = discord.app_commands.Group(name="gemini", description="Geminiモデル関連の操作を行います。")

# --- スラッシュコマンド ---

@bot.tree.command(name="upload_guide", description="ファイルアップロード方法の案内") # このコマンドは残す
async def upload_guide(interaction: discord.Interaction):
    await interaction.response.send_message(
        "ファイルをアップロードするには、このチャンネルに直接ファイルをドラッグ＆ドロップするか、メッセージ入力欄の「+」ボタンからファイルを添付して送信してください。\n"
        "画像または動画ファイルが対象です。", ephemeral=True)

# --- /files サブコマンド群は全て削除 ---

# --- /gemini サブコマンド ---
@gemini_group.command(name="list", description="自動タグ付けに利用可能なGeminiモデルの一覧を表示します。")
async def gemini_list(interaction: discord.Interaction):
    if not GEMINI_API_KEY: await interaction.response.send_message("Gemini APIキーが設定されていません。", ephemeral=True); return
    if not gemini_model_instance: await interaction.response.send_message("Geminiモデルが初期化されていません。", ephemeral=True); return
    await interaction.response.defer(ephemeral=True)
    try:
        models_info = "利用可能なGeminiモデル (generateContentサポート):\n"; count = 0
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                model_display_name = model.name.replace("models/", "")
                models_info += f"- `{model_display_name}` ({model.display_name})\n"; count += 1
                if len(models_info) > 1800: await interaction.followup.send(models_info, ephemeral=True); models_info = ""
        if count == 0: models_info = "利用可能なGeminiモデルが見つかりませんでした。"
        if models_info: await interaction.followup.send(models_info, ephemeral=True)
    except Exception as e: await interaction.followup.send(f"モデル一覧の取得中にエラー: {e}", ephemeral=True)

@gemini_group.command(name="set", description="自動タグ付けに使用するGeminiモデルを設定します。")
@discord.app_commands.describe(model_name="Geminiモデル名 (例: gemini-1.5-flash-latest)。")
@discord.app_commands.autocomplete(model_name=gemini_model_autocomplete)
@is_admin()
async def gemini_set(interaction: discord.Interaction, model_name: str):
    global current_gemini_model, gemini_model_instance
    if not GEMINI_API_KEY: await interaction.response.send_message("Gemini APIキーが設定されていません。", ephemeral=True); return
    if not gemini_model_instance: await interaction.response.send_message("Geminiモデルが初期化されていません。", ephemeral=True); return
    await interaction.response.defer(ephemeral=True)
    try:
        full_model_name_to_check = model_name if model_name.startswith("models/") else f"models/{model_name}"
        retrieved_model = genai.get_model(full_model_name_to_check)
        if 'generateContent' not in retrieved_model.supported_generation_methods:
            await interaction.followup.send(f"モデル `{model_name}` は `generateContent` をサポートしていません。", ephemeral=True); return
        new_model_instance = genai.GenerativeModel(retrieved_model.name, safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE })
        current_gemini_model = retrieved_model.name.replace("models/", "")
        gemini_model_instance = new_model_instance
        await interaction.followup.send(f"自動タグ付けのGeminiモデルを `{current_gemini_model}` に設定しました。", ephemeral=True)
        print(f"Geminiモデルが '{current_gemini_model}' に変更されました。")
    except Exception as e:
        await interaction.followup.send(f"モデル `{model_name}` の設定に失敗: {e}", ephemeral=True)
        print(f"Geminiモデル '{model_name}' の設定失敗: {e}")

@gemini_group.command(name="current", description="現在設定されているGeminiモデル名を表示します。")
async def gemini_current(interaction: discord.Interaction):
    if not gemini_model_instance:
        await interaction.response.send_message(f"Geminiモデルは現在設定されていません、または初期化に失敗しています。", ephemeral=True)
    else: await interaction.response.send_message(f"現在設定されているGeminiモデルは `{current_gemini_model}` です。", ephemeral=True)

@bot.tree.command(name="help_nasbot", description="このBOTのコマンド一覧と簡単な説明を表示します。")
async def help_nasbot(interaction: discord.Interaction):
    embed = discord.Embed(title="ファイル管理BOT ヘルプ", description="このBOTで利用可能なコマンド一覧です。", color=discord.Color.blue())
    # /files グループに関する記述を削除
    embed.add_field(name="Geminiモデル設定 (`/gemini`)", value=(
        "`  set <model_name <モデル名>]` - (管理者) 自動タグ付けに使用するGeminiモデルを設定します。\n"
        "`  current` - 現在のGeminiモデル名を表示します。\n"
        "`  list` - 利用可能なGeminiモデルの一覧を表示します。\n"
    ), inline=False)
    embed.add_field(name="その他", value=(
        "`/upload_guide` - ファイルのアップロード方法を表示します。\n"
        "`/help_nasbot` - このヘルプを表示します。"
    ), inline=False)
    embed.set_footer(text="ファイルを直接このチャンネルにアップロードすることでも処理が開始されます。")
    await interaction.response.send_message(embed=embed, ephemeral=True)

# --- コマンドグループをBOTに追加 ---
# bot.tree.add_command(files_group) # /files グループの登録を削除
bot.tree.add_command(gemini_group)

# --- BOT実行 ---
if __name__ == "__main__":
    if DISCORD_BOT_TOKEN:
        if not GEMINI_API_KEY:
            print("警告: GEMINI_API_KEYが .envファイルに設定されていません。Gemini API関連の機能は利用できません。")
        bot.run(DISCORD_BOT_TOKEN)
    else:
        print("エラー: DISCORD_BOT_TOKEN が .envファイルに設定されていません。")