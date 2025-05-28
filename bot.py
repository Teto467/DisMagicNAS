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

# --- 設定 ---
load_dotenv()
DISCORD_BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
BASE_UPLOAD_FOLDER = os.getenv('BASE_UPLOAD_FOLDER', 'uploads')
ADMIN_ROLE_NAMES_STR = os.getenv('ADMIN_ROLE_NAMES', 'BOT管理者,運営スタッフ')
ADMIN_ROLE_NAMES = [name.strip() for name in ADMIN_ROLE_NAMES_STR.split(',')]
DEFAULT_GEMINI_MODEL = os.getenv('DEFAULT_GEMINI_MODEL', 'gemini-1.5-flash-latest')

# --- タグ付けプロンプトファイル名定義 ---
TAGGING_PROMPT_FILE = "Tagging_prompt.txt"

# --- タグ付けプロンプト読み込み関数 ---
def load_tagging_prompt():
    if os.path.exists(TAGGING_PROMPT_FILE):
        try:
            with open(TAGGING_PROMPT_FILE, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
                if prompt:
                    print(f"タグ付けプロンプトを '{TAGGING_PROMPT_FILE}' から読み込みました。")
                    return prompt
                else:
                    print(f"警告: '{TAGGING_PROMPT_FILE}' は空です。デフォルトのプロンプトを使用します。")
        except Exception as e:
            print(f"警告: '{TAGGING_PROMPT_FILE}' の読み込みに失敗しました: {e}。デフォルトのプロンプトを使用します。")
    else:
        print(f"情報: '{TAGGING_PROMPT_FILE}' が見つかりません。デフォルトのプロンプトを使用します。")
    return DEFAULT_TAGGING_PROMPT

# --- Gemini API 初期化 ---
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

# --- BOT 初期化 ---
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix='/', intents=intents)

# --- ヘルパー関数 ---
def sanitize_filename_component(text):
    return re.sub(r'[\\/*?:"<>|\s]', '_', text)

def get_file_icon(extension):
    ext = extension.lower()
    if ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']:
        return "🖼️"
    elif ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
        return "🎬"
    elif ext in ['.txt', '.md', '.doc', '.pdf']:
        return "📄"
    else:
        return "📁"

def create_year_month_folder_if_not_exists(base_folder):
    now = datetime.datetime.now()
    year_month_folder_name = now.strftime("%Y%m")
    year_month_folder_path = os.path.join(base_folder, year_month_folder_name)
    if not os.path.exists(year_month_folder_path):
        os.makedirs(year_month_folder_path)
        print(f"年月フォルダ '{year_month_folder_path}' を作成しました。")
    return year_month_folder_path

# --- 管理者チェック ---
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

# --- Gemini API 関連 ---
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
        if 'uploaded_file' in locals() and uploaded_file:
             try:
                 # genai.delete_file(uploaded_file.name) # 同期版の場合
                 pass
             except Exception as e_del:
                 print(f"Gemini APIからアップロードされたファイル {uploaded_file.name} の削除中にエラー: {e_del}")

# --- イベントハンドラ ---
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
async def on_message(message):
    if message.author == bot.user:
        return
    if message.attachments:
        year_month_folder_path = create_year_month_folder_if_not_exists(BASE_UPLOAD_FOLDER)
        ctx = await bot.get_context(message)
        for attachment in message.attachments:
            allowed_image_types = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp')
            allowed_video_types = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
            file_ext = os.path.splitext(attachment.filename)[1].lower()
            if not (file_ext in allowed_image_types or file_ext in allowed_video_types):
                await message.channel.send(
                    f"ファイル '{attachment.filename}' の形式 ({file_ext}) はサポートされていません。\n"
                    f"対応形式 (画像): {', '.join(allowed_image_types)}\n"
                    f"対応形式 (動画): {', '.join(allowed_video_types)}"
                )
                continue
            if attachment.size > 8 * 1024 * 1024 and not (ctx.guild and ctx.guild.premium_tier >= 1):
                await message.channel.send(
                    f"ファイル '{attachment.filename}' ({attachment.size // 1024 // 1024}MB) はサイズが大きすぎます。"
                    "サーバーブーストレベルに応じて上限が緩和されますが、基本は8MBまでです。"
                )
                continue
            temp_save_path = os.path.join(year_month_folder_path, f"temp_{attachment.filename}")
            await attachment.save(temp_save_path)
            processing_msg = await message.channel.send(f"ファイル '{attachment.filename}' をアップロード中... 自動タグ付け処理を開始します。しばらくお待ちください。")
            tags_str = "notags"
            if gemini_model_instance:
                try:
                    if file_ext in allowed_image_types:
                        try:
                            img = Image.open(temp_save_path)
                            img.verify()
                            img.close()
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
                await processing_msg.edit(content=(
                    f"ファイル '{attachment.filename}' をアップロードし、'{new_filename}' として保存しました。\n"
                    f"自動タグ: `{tags_str if tags_str != 'notags' else 'なし'}`"
                ))
            except Exception as e:
                print(f"ファイルのリネーム/保存中にエラー: {e}")
                await processing_msg.edit(content=f"ファイル '{attachment.filename}' の最終保存中にエラーが発生しました。")
                if os.path.exists(temp_save_path):
                    os.remove(temp_save_path)
    await bot.process_commands(message)

# --- ★ オートコンプリート用の関数 ---
async def year_month_autocomplete(interaction: discord.Interaction, current: str) -> list[discord.app_commands.Choice[str]]:
    choices = []
    if os.path.exists(BASE_UPLOAD_FOLDER):
        for item in os.listdir(BASE_UPLOAD_FOLDER):
            item_path = os.path.join(BASE_UPLOAD_FOLDER, item)
            if os.path.isdir(item_path) and re.fullmatch(r"\d{6}", item):
                if current.lower() in item.lower():
                    choices.append(discord.app_commands.Choice(name=item, value=item))
    return choices[:25] # Discordのオートコンプリートの選択肢上限は25個

async def filename_autocomplete(interaction: discord.Interaction, current: str) -> list[discord.app_commands.Choice[str]]:
    choices = []
    # コマンドのオプションから year_month を取得しようと試みる
    # interaction.data は生のインタラクションデータを含む辞書
    year_month_input = None
    if interaction.data and 'options' in interaction.data:
        for option in interaction.data['options']:
            if option['name'] == 'year_month' and 'value' in option : # year_monthが入力されているか確認
                year_month_input = option['value']
                break # 通常、同じ名前のオプションは1つのはず
            # ネストされたコマンドの場合、さらに深く探索する必要があるかもしれない
            elif 'options' in option: # サブコマンドやグループの場合
                 for sub_option in option['options']:
                    if sub_option['name'] == 'year_month' and 'value' in sub_option:
                        year_month_input = sub_option['value']
                        break
                 if year_month_input:
                     break


    search_folders = []
    if year_month_input and re.fullmatch(r"\d{6}", str(year_month_input)):
        target_folder = os.path.join(BASE_UPLOAD_FOLDER, str(year_month_input))
        if os.path.exists(target_folder) and os.path.isdir(target_folder):
            search_folders.append(target_folder)
    elif not year_month_input: # year_month が指定されていない場合は全フォルダを検索（負荷注意）
        if os.path.exists(BASE_UPLOAD_FOLDER):
            for item in os.listdir(BASE_UPLOAD_FOLDER):
                item_path = os.path.join(BASE_UPLOAD_FOLDER, item)
                if os.path.isdir(item_path) and re.fullmatch(r"\d{6}", item):
                    search_folders.append(item_path)
    
    # current が空の場合は候補を出さないか、あるいは人気ファイルなどを出す（今回は空なら出さない）
    if not current and not choices: # currentが空なら何もしない（ファイルが多すぎるため）
        return []

    files_found = []
    for folder in search_folders:
        for filename in os.listdir(folder):
            if os.path.isfile(os.path.join(folder, filename)):
                if current.lower() in filename.lower():
                    files_found.append(discord.app_commands.Choice(name=filename, value=filename))
            if len(files_found) >= 25: # 候補が25件に達したら終了
                break
        if len(files_found) >= 25:
            break
    return files_found

async def gemini_model_autocomplete(interaction: discord.Interaction, current: str) -> list[discord.app_commands.Choice[str]]:
    choices = []
    # 修正: APIキーの存在とモデルインスタンスの初期化状態で判断
    if not GEMINI_API_KEY or not gemini_model_instance:
        return []
    try:
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                model_display_name = model.name.replace("models/", "")
                if current.lower() in model_display_name.lower():
                    # Choiceのnameが長すぎるとエラーになることがあるため調整
                    choice_name = f"{model_display_name} ({model.display_name})"
                    if len(choice_name) > 100: # DiscordのChoice名の制限は100文字
                        choice_name = model_display_name[:97] + "..." if len(model_display_name) > 97 else model_display_name

                    choices.append(discord.app_commands.Choice(name=choice_name, value=model_display_name))
            if len(choices) >= 25:
                break
    except Exception as e:
        print(f"Geminiモデルのオートコンプリート中にエラー: {e}")
    return choices
# --- スラッシュコマンド ---
@bot.tree.command(name="upload_guide", description="ファイルアップロード方法の案内")
async def upload_guide(interaction: discord.Interaction):
    await interaction.response.send_message(
        "ファイルをアップロードするには、このチャンネルに直接ファイルをドラッグ＆ドロップするか、メッセージ入力欄の「+」ボタンからファイルを添付して送信してください。\n"
        "画像または動画ファイルが対象です。",
        ephemeral=True
    )

@bot.tree.command(name="list_files", description="保存されているファイルの一覧を表示します。年月やキーワードで絞り込み可能。")
@discord.app_commands.describe(year_month="表示したい年月 (例: 202505)。省略すると全期間。", keyword="ファイル名に含まれる検索キーワード。")
@discord.app_commands.autocomplete(year_month=year_month_autocomplete) # ★ year_month オートコンプリート追加
async def list_files(interaction: discord.Interaction, year_month: str = None, keyword: str = None):
    await interaction.response.defer(ephemeral=True)
    found_files = []
    search_folders = []
    if year_month:
        if not re.fullmatch(r"\d{6}", year_month):
            await interaction.followup.send("年月の形式が正しくありません。`YYYYMM` (例: `202505`) の形式で入力してください。")
            return
        target_folder = os.path.join(BASE_UPLOAD_FOLDER, year_month)
        if os.path.exists(target_folder) and os.path.isdir(target_folder):
            search_folders.append(target_folder)
        else:
            await interaction.followup.send(f"`{year_month}` に該当するファイルは見つかりませんでした。")
            return
    else:
        if os.path.exists(BASE_UPLOAD_FOLDER):
            for item in os.listdir(BASE_UPLOAD_FOLDER):
                item_path = os.path.join(BASE_UPLOAD_FOLDER, item)
                if os.path.isdir(item_path) and re.fullmatch(r"\d{6}", item):
                    search_folders.append(item_path)
    if not search_folders and not year_month:
        await interaction.followup.send("まだアップロードされたファイルはありません。")
        return
    elif not search_folders and year_month:
         await interaction.followup.send(f"`{year_month}` に該当するファイルは見つかりませんでした。")
         return
    for folder in search_folders:
        for filename in os.listdir(folder):
            if os.path.isfile(os.path.join(folder, filename)):
                if keyword:
                    if keyword.lower() in filename.lower():
                        found_files.append(filename)
                else:
                    found_files.append(filename)
    if not found_files:
        msg = "該当するファイルは見つかりませんでした。"
        if keyword: msg += f" (キーワード: `{keyword}`)"
        if year_month: msg += f" (年月: `{year_month}`)"
        await interaction.followup.send(msg)
        return
    response_message = f"ファイル一覧 ({len(found_files)}件):\n"
    if keyword: response_message += f"検索キーワード: `{keyword}`\n"
    if year_month: response_message += f"年月: `{year_month}`\n"
    response_message += "```\n"
    current_length = len(response_message)
    files_in_chunk = 0
    for filename in sorted(found_files):
        file_ext = os.path.splitext(filename)[1]
        icon = get_file_icon(file_ext)
        line = f"{icon} {filename}\n"
        if current_length + len(line) > 1980:
            response_message += "```"
            await interaction.followup.send(response_message)
            response_message = "```\n" + line
            current_length = len(response_message)
            files_in_chunk = 1
        else:
            response_message += line
            current_length += len(line)
            files_in_chunk +=1
    if files_in_chunk > 0:
        response_message += "```"
        await interaction.followup.send(response_message)

@bot.tree.command(name="search_files", description="ファイル名やタグでファイルを検索します。")
@discord.app_commands.describe(keyword="検索キーワード (ファイル名、日付、タグの一部など)。")
async def search_files(interaction: discord.Interaction, keyword: str):
    if not keyword or len(keyword) < 2 :
        await interaction.response.send_message("検索キーワードを2文字以上で入力してください。", ephemeral=True)
        return
    # list_filesコマンドに処理を委譲（実質同じ機能なので）
    # 修正: self=bot を削除
    await list_files.callback(interaction=interaction, year_month=None, keyword=keyword) # type: ignore [attr-defined]

@bot.tree.command(name="download_file", description="指定されたファイル名のファイルをダウンロードします。")
@discord.app_commands.describe(
    year_month="ファイルが存在する年月 (例: 202505)。省略すると全フォルダ検索。", # ★ 引数追加
    filename="ダウンロードしたい正確なファイル名。"
)
@discord.app_commands.autocomplete(year_month=year_month_autocomplete, filename=filename_autocomplete) # ★ オートコンプリート追加
async def download_file(interaction: discord.Interaction, filename: str, year_month: str = None): # ★ year_month引数追加
    await interaction.response.defer(ephemeral=False)
    found_path = None
    search_folders = []

    if year_month:
        if not re.fullmatch(r"\d{6}", year_month):
            await interaction.followup.send("年月の形式が正しくありません。`YYYYMM`の形式で入力してください。", ephemeral=True)
            return
        target_folder = os.path.join(BASE_UPLOAD_FOLDER, year_month)
        if os.path.exists(target_folder) and os.path.isdir(target_folder):
            search_folders.append(target_folder)
        else:
            await interaction.followup.send(f"指定された年月フォルダ `{year_month}` が見つかりません。", ephemeral=True)
            return
    else: # year_monthが指定されていない場合は全年月フォルダを対象
        if os.path.exists(BASE_UPLOAD_FOLDER):
            for item in os.listdir(BASE_UPLOAD_FOLDER):
                item_path = os.path.join(BASE_UPLOAD_FOLDER, item)
                if os.path.isdir(item_path) and re.fullmatch(r"\d{6}", item):
                    search_folders.append(item_path)
    
    if not search_folders:
        await interaction.followup.send("検索対象のフォルダが見つかりません。", ephemeral=True)
        return

    for folder_path in search_folders:
        prospective_path = os.path.join(folder_path, filename)
        if os.path.exists(prospective_path) and os.path.isfile(prospective_path):
            found_path = prospective_path
            break
    
    if found_path:
        try:
            file_size = os.path.getsize(found_path)
            if file_size > 8 * 1024 * 1024: # 8MB
                await interaction.followup.send(
                    f"ファイル '{filename}' ({file_size // 1024 // 1024}MB) はサイズが大きすぎるため、直接送信できません。"
                )
                return
            await interaction.followup.send(f"ファイル '{filename}' を送信します:", file=discord.File(found_path))
        except Exception as e:
            print(f"ファイル送信エラー: {e}")
            await interaction.followup.send(f"ファイル '{filename}' の送信中にエラーが発生しました。")
    else:
        await interaction.followup.send(f"ファイル '{filename}' が見つかりませんでした。ファイル名と年月を確認してください。")

@bot.tree.command(name="edit_tags", description="ファイルのタグを編集します（ファイル名リネーム）。")
@discord.app_commands.describe(
    year_month="ファイルが存在する年月 (例: 202505)。", # ★ 引数追加
    current_filename="現在の完全なファイル名。",
    new_tags="新しいタグ (カンマ区切り、例: タグ1,タグ2,タグ3)。タグなしは notags と入力。"
)
@discord.app_commands.autocomplete(year_month=year_month_autocomplete, current_filename=filename_autocomplete) # ★ オートコンプリート追加
@is_admin()
async def edit_tags(interaction: discord.Interaction, year_month: str, current_filename: str, new_tags: str): # ★ year_month引数追加 (必須とした)
    await interaction.response.defer(ephemeral=True)

    if not re.fullmatch(r"\d{6}", year_month):
        await interaction.followup.send("年月の形式が正しくありません。`YYYYMM`の形式で入力してください。")
        return

    original_ym_folder_path = os.path.join(BASE_UPLOAD_FOLDER, year_month)
    if not (os.path.exists(original_ym_folder_path) and os.path.isdir(original_ym_folder_path)):
        await interaction.followup.send(f"指定された年月フォルダ `{year_month}` が見つかりません。")
        return

    current_filepath = os.path.join(original_ym_folder_path, current_filename)
    if not (os.path.exists(current_filepath) and os.path.isfile(current_filepath)):
        await interaction.followup.send(f"ファイル '{current_filename}' が年月フォルダ `{year_month}` 内に見つかりませんでした。")
        return
    
    if new_tags.strip().lower() == "notags":
        processed_new_tags = "notags"
    else:
        tags_list = [sanitize_filename_component(tag.strip()) for tag in new_tags.split(',') if tag.strip()]
        if not tags_list:
            await interaction.followup.send("新しいタグが指定されていません。タグなしにする場合は `notags` と入力してください。")
            return
        processed_new_tags = "-".join(tags_list)
    
    parts = current_filename.split('_', 2)
    if len(parts) < 3:
        await interaction.followup.send(f"ファイル '{current_filename}' は期待される命名規則に従っていません。手動でのリネームが必要かもしれません。")
        return
    date_str = parts[0]
    original_name_with_ext = parts[2]
    base, ext = os.path.splitext(original_name_with_ext)
    new_filename_constructed = f"{date_str}_{processed_new_tags}_{base}{ext}"
    new_filepath = os.path.join(original_ym_folder_path, new_filename_constructed)

    if current_filepath == new_filepath:
        await interaction.followup.send(f"新しいタグは現在のタグと同じです。変更はありませんでした。")
        return
    if os.path.exists(new_filepath):
        await interaction.followup.send(f"エラー: 新しいファイル名 '{new_filename_constructed}' は既に存在します。")
        return
    try:
        os.rename(current_filepath, new_filepath)
        await interaction.followup.send(
            f"ファイル '{current_filename}' のタグを編集しました。\n"
            f"新しいファイル名: `{new_filename_constructed}`"
        )
        print(f"ファイル名変更: '{current_filename}' -> '{new_filename_constructed}'")
    except Exception as e:
        print(f"ファイル名変更エラー: {e}")
        await interaction.followup.send(f"タグの編集中にエラーが発生しました: {e}")

@bot.tree.command(name="delete_file", description="指定されたファイルを削除します。")
@discord.app_commands.describe(
    year_month="ファイルが存在する年月 (例: 202505)。", # ★ 引数追加
    filename="削除したい正確なファイル名。"
)
@discord.app_commands.autocomplete(year_month=year_month_autocomplete, filename=filename_autocomplete) # ★ オートコンプリート追加
@is_admin()
async def delete_file(interaction: discord.Interaction, year_month: str, filename: str): # ★ year_month引数追加 (必須とした)
    await interaction.response.defer(ephemeral=True)

    if not re.fullmatch(r"\d{6}", year_month):
        await interaction.followup.send("年月の形式が正しくありません。`YYYYMM`の形式で入力してください。")
        return

    target_ym_folder_path = os.path.join(BASE_UPLOAD_FOLDER, year_month)
    if not (os.path.exists(target_ym_folder_path) and os.path.isdir(target_ym_folder_path)):
        await interaction.followup.send(f"指定された年月フォルダ `{year_month}` が見つかりません。")
        return
        
    found_path = os.path.join(target_ym_folder_path, filename)
    if os.path.exists(found_path) and os.path.isfile(found_path):
        view = ConfirmDeleteView(found_path, filename, interaction.user)
        await interaction.followup.send(
            f"本当にファイル `{filename}` (フォルダ: `{year_month}`) を削除しますか？この操作は取り消せません。",
            view=view,
            ephemeral=True
        )
        await view.wait()
    else:
        await interaction.followup.send(f"ファイル '{filename}' が年月フォルダ `{year_month}` 内に見つかりませんでした。")

class ConfirmDeleteView(discord.ui.View):
    def __init__(self, filepath: str, filename: str, author: discord.User, timeout=30.0):
        super().__init__(timeout=timeout)
        self.filepath = filepath
        self.filename = filename
        self.author = author
    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.author.id:
            await interaction.response.send_message("この操作はコマンドを実行した本人のみが行えます。", ephemeral=True)
            return False
        return True
    @discord.ui.button(label="はい、削除します", style=discord.ButtonStyle.danger, custom_id="confirm_delete")
    async def confirm_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        try:
            os.remove(self.filepath)
            print(f"ファイル '{self.filename}' をユーザー '{interaction.user}' の指示により削除しました。")
            await interaction.response.edit_message(content=f"ファイル `{self.filename}` を削除しました。", view=None)
        except Exception as e:
            print(f"ファイル削除エラー ({self.filename}): {e}")
            await interaction.response.edit_message(content=f"ファイル `{self.filename}` の削除中にエラーが発生しました。", view=None)
        self.stop()
    @discord.ui.button(label="いいえ、キャンセル", style=discord.ButtonStyle.secondary, custom_id="cancel_delete")
    async def cancel_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.edit_message(content="ファイル削除をキャンセルしました。", view=None)
        self.stop()
    async def on_timeout(self):
        print(f"ファイル '{self.filename}' の削除確認がタイムアウトしました。")
        pass

@bot.tree.command(name="list_gemini_models", description="自動タグ付けに利用可能なGeminiモデルの一覧を表示します。")
async def list_gemini_models(interaction: discord.Interaction):
    # 修正: APIキーの存在とモデルインスタンスの初期化状態で判断
    if not GEMINI_API_KEY:
        await interaction.response.send_message("Gemini APIキーが設定されていません。", ephemeral=True)
        return
    if not gemini_model_instance:
        await interaction.response.send_message("Geminiモデルが初期化されていません。APIキーを確認してください。", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True)
    try:
        models_info = "利用可能なGeminiモデル (generateContentサポート):\n"
        count = 0
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                model_display_name = model.name.replace("models/", "")
                models_info += f"- `{model_display_name}` ({model.display_name})\n"
                count += 1
                if len(models_info) > 1800:
                    await interaction.followup.send(models_info, ephemeral=True)
                    models_info = ""
        if count == 0: models_info = "利用可能なGeminiモデルが見つかりませんでした (generateContentサポート)。"
        if models_info: await interaction.followup.send(models_info, ephemeral=True)
    except Exception as e:
        await interaction.followup.send(f"モデル一覧の取得中にエラーが発生しました: {e}", ephemeral=True)

@bot.tree.command(name="set_model", description="自動タグ付けに使用するGeminiモデルを設定します。")
@discord.app_commands.describe(model_name="Geminiモデル名 (例: gemini-1.5-flash-latest)。")
@discord.app_commands.autocomplete(model_name=gemini_model_autocomplete)
@is_admin()
async def set_model(interaction: discord.Interaction, model_name: str):
    global current_gemini_model, gemini_model_instance
    # 修正: APIキーの存在とモデルインスタンスの初期化状態で判断
    if not GEMINI_API_KEY:
        await interaction.response.send_message("Gemini APIキーが設定されていません。", ephemeral=True)
        return
    if not gemini_model_instance: # 初期化に失敗している場合もここで捉える
         await interaction.response.send_message("Geminiモデルが初期化されていません。APIキーやデフォルトモデル名を確認してください。", ephemeral=True)
         return

    await interaction.response.defer(ephemeral=True)
    try:
        full_model_name_to_check = model_name if model_name.startswith("models/") else f"models/{model_name}"
        retrieved_model = genai.get_model(full_model_name_to_check) # これが失敗すればモデルは存在しない
        if 'generateContent' not in retrieved_model.supported_generation_methods:
            await interaction.followup.send(
                f"モデル `{model_name}` は `generateContent` をサポートしていません。\n"
                "`/list_gemini_models` でサポートされているモデルを確認してください。",
                ephemeral=True)
            return
        # 新しいモデルでインスタンスを再作成する前に、現在のインスタンスを破棄する処理は通常不要
        # genai.GenerativeModel() で新しいものを作ればよい
        new_model_instance = genai.GenerativeModel(
            retrieved_model.name,
            safety_settings={HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                             HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                             HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                             HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE})
        current_gemini_model = retrieved_model.name.replace("models/", "")
        gemini_model_instance = new_model_instance # グローバルインスタンスを更新
        await interaction.followup.send(f"自動タグ付けに使用するGeminiモデルを `{current_gemini_model}` に設定しました。", ephemeral=True)
        print(f"Geminiモデルが '{current_gemini_model}' に変更されました。")
    except Exception as e:
        error_message = f"モデル `{model_name}` の設定に失敗しました: {e}\n"
        error_message += "入力されたモデル名が正しいか、`/list_gemini_models` コマンドで利用可能なモデルを確認してください。"
        await interaction.followup.send(error_message, ephemeral=True)
        print(f"Geminiモデル '{model_name}' の設定失敗: {e}")

@bot.tree.command(name="current_model", description="現在設定されているGeminiモデル名を表示します。")
async def current_model(interaction: discord.Interaction):
    if not gemini_model_instance:
        await interaction.response.send_message(f"Geminiモデルは現在設定されていません、または初期化に失敗しています。", ephemeral=True)
    else:
        await interaction.response.send_message(f"現在設定されているGeminiモデルは `{current_gemini_model}` です。", ephemeral=True)

@bot.tree.command(name="help_nasbot", description="このBOTのコマンド一覧と簡単な説明を表示します。")
async def help_nasbot(interaction: discord.Interaction):
    embed = discord.Embed(title="ファイル管理BOT ヘルプ", description="このBOTで利用可能なコマンド一覧です。", color=discord.Color.blue())
    embed.add_field(name="ファイル操作", value=(
        "`/upload_guide` - ファイルのアップロード方法を表示します。\n"
        "`/list_files [年月] [キーワード]` - ファイル一覧を表示します。\n"
        "`/search_files <キーワード>` - ファイルを検索します。\n"
        "`/download_file [年月] <ファイル名>` - ファイルをダウンロードします。" # ★ 変更
    ), inline=False)
    embed.add_field(name="管理者向けコマンド", value=(
        "`/edit_tags <年月> <現在のファイル名> <新しいタグ>` - ファイルのタグを編集します。\n" # ★ 変更
        "`/delete_file <年月> <ファイル名>` - ファイルを削除します。\n" # ★ 変更
        "`/set_model <モデル名>` - 自動タグ付けに使用するGeminiモデルを設定します。"
    ), inline=False)
    embed.add_field(name="その他", value=(
        "`/current_model` - 現在のGeminiモデル名を表示します。\n"
        "`/list_gemini_models` - 利用可能なGeminiモデルの一覧を表示します。\n"
        "`/help_nasbot` - このヘルプを表示します。"
    ), inline=False)
    embed.set_footer(text="ファイルを直接このチャンネルにアップロードすることでも処理が開始されます。")
    await interaction.response.send_message(embed=embed, ephemeral=True)

# --- BOT実行 ---
if __name__ == "__main__":
    if DISCORD_BOT_TOKEN:
        if not GEMINI_API_KEY:
            print("警告: GEMINI_API_KEYが .envファイルに設定されていません。Gemini API関連の機能は利用できません。")
        bot.run(DISCORD_BOT_TOKEN)
    else:
        print("エラー: DISCORD_BOT_TOKEN が .envファイルに設定されていません。")