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
from discord import app_commands

# Google Drive API 関連のインポート
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    google_drive_libs_available = True
except ImportError:
    google_drive_libs_available = False
    print("警告: Google Drive関連のライブラリが見つかりません。`pip install google-api-python-client google-auth-httplib2 google-auth` を実行してください。")


# --- 設定ファイル名 ---
CONFIG_FILE_NAME = "config.json"

# --- デフォルト設定 ---
DEFAULT_CONFIG = {
    "admin_role_names": ["BOT管理者", "運営スタッフ"], # Geminiコマンドおよびアップロード設定コマンドの管理ロール
    "default_gemini_model": "gemini-1.5-flash-latest",
    "tagging_prompt_file": "Tagging_prompt.txt",
    "base_upload_folder": "uploads",
    "upload_destination": "local",   # "local" or "gdrive"
    "gdrive_service_account_key_path": "service-account-key.json",
    "gdrive_target_folder_id": None,
    "gdrive_create_ym_folders": True
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

admin_roles_from_config = bot_config.get("admin_role_names")
if isinstance(admin_roles_from_config, list):
    ADMIN_ROLE_NAMES = admin_roles_from_config
else:
    ADMIN_ROLE_NAMES = DEFAULT_CONFIG["admin_role_names"]
    if admin_roles_from_config is not None:
        print(f"警告: '{CONFIG_FILE_NAME}' の 'admin_role_names' がリスト形式ではありません。デフォルト設定 ({ADMIN_ROLE_NAMES}) を使用します。")

DEFAULT_GEMINI_MODEL = bot_config.get("default_gemini_model", DEFAULT_CONFIG["default_gemini_model"])
TAGGING_PROMPT_FILE = bot_config.get("tagging_prompt_file", DEFAULT_CONFIG["tagging_prompt_file"])
BASE_UPLOAD_FOLDER = bot_config.get("base_upload_folder", DEFAULT_CONFIG["base_upload_folder"])

UPLOAD_DESTINATION = bot_config.get("upload_destination", DEFAULT_CONFIG["upload_destination"])
GDRIVE_SERVICE_ACCOUNT_KEY_PATH = bot_config.get("gdrive_service_account_key_path", DEFAULT_CONFIG["gdrive_service_account_key_path"])
GDRIVE_TARGET_FOLDER_ID = bot_config.get("gdrive_target_folder_id")
GDRIVE_CREATE_YM_FOLDERS = bot_config.get("gdrive_create_ym_folders", DEFAULT_CONFIG["gdrive_create_ym_folders"])

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
                else: print(f"警告: '{prompt_file_path}' は空です。デフォルトのプロンプトを使用します。")
        except Exception as e: print(f"警告: '{prompt_file_path}' の読み込みに失敗しました: {e}。デフォルトのプロンプトを使用します。")
    else: print(f"情報: '{prompt_file_path}' が見つかりません。デフォルトのプロンプトを使用します。")
    return DEFAULT_TAGGING_PROMPT_TEXT

gemini_model_instance = None
current_gemini_model = DEFAULT_GEMINI_MODEL
gdrive_service = None

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    try:
        gemini_model_instance = genai.GenerativeModel(
            current_gemini_model,
            safety_settings={ HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                             HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                             HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                             HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,})
        print(f"Geminiモデル '{current_gemini_model}' の初期化に成功しました。")
    except Exception as e:
        print(f"エラー: デフォルトのGeminiモデル '{current_gemini_model}' の初期化に失敗しました: {e}")
        gemini_model_instance = None
else: print("情報: GEMINI_API_KEYが設定されていません。Gemini API関連の機能は利用できません。")

def initialize_gdrive_service():
    global gdrive_service, google_drive_libs_available
    if not google_drive_libs_available:
        gdrive_service = None
        print("Google Drive機能はライブラリが不足しているため無効です。")
        return

    creds_path = GDRIVE_SERVICE_ACCOUNT_KEY_PATH
    if not creds_path or not os.path.exists(creds_path):
        print(f"情報: Google Driveのサービスアカウントキーファイルが見つかりません: {creds_path}。Drive機能は無効です。")
        gdrive_service = None
        return
    try:
        scopes = ['https://www.googleapis.com/auth/drive']
        creds = service_account.Credentials.from_service_account_file(creds_path, scopes=scopes)
        gdrive_service = build('drive', 'v3', credentials=creds, cache_discovery=False)
        print("Google Driveサービスが正常に初期化されました。")
    except Exception as e:
        print(f"Google Driveサービスの初期化に失敗しました: {e}")
        gdrive_service = None

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix='/', intents=intents)

# --- ヘルパー関数 ---
def sanitize_filename_component(text): return re.sub(r'[\\/*?:"<>|\s]', '_', text)
def get_file_icon(extension): # 現在未使用
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
        print(f"ローカル年月フォルダ '{year_month_folder_path}' を作成しました。")
    return year_month_folder_path
def parse_bot_filename(filename_str: str) -> dict:
    parts = {"date": "不明", "tags_raw": "notags", "tags_display": "タグなし", "original_stem": filename_str, "extension": ""}
    base_name, ext = os.path.splitext(filename_str)
    parts["extension"] = ext
    match = re.match(r"(\d{8})_([^_]+)_(.+)", base_name)
    if match:
        parts["date"], parts["tags_raw"], parts["original_stem"] = match.groups()
        if parts["tags_raw"] == "notags": parts["tags_display"] = "タグなし"
        else: parts["tags_display"] = parts["tags_raw"].replace("_", "-")
    else:
        match_no_tags = re.match(r"(\d{8})_(.+)", base_name)
        if match_no_tags: parts["date"], parts["original_stem"] = match_no_tags.groups()
        else: parts["original_stem"] = base_name
    return parts

def save_bot_config(new_settings: dict):
    global bot_config, UPLOAD_DESTINATION, GDRIVE_TARGET_FOLDER_ID, GDRIVE_CREATE_YM_FOLDERS, GDRIVE_SERVICE_ACCOUNT_KEY_PATH
    current_full_config = {}
    if os.path.exists(CONFIG_FILE_NAME):
        try:
            with open(CONFIG_FILE_NAME, "r", encoding="utf-8") as f: current_full_config = json.load(f)
        except Exception as e:
            print(f"config.json の読み込み中にエラーが発生したため、更新は現在のメモリ上の設定をベースにします: {e}")
            current_full_config = bot_config.copy()
    else: current_full_config = bot_config.copy()
    current_full_config.update(new_settings)
    try:
        with open(CONFIG_FILE_NAME, "w", encoding="utf-8") as f:
            json.dump(current_full_config, f, indent=4, ensure_ascii=False)
        print(f"設定を '{CONFIG_FILE_NAME}' に保存しました。")
        bot_config.update(new_settings)
        UPLOAD_DESTINATION = bot_config.get("upload_destination", DEFAULT_CONFIG["upload_destination"])
        GDRIVE_TARGET_FOLDER_ID = bot_config.get("gdrive_target_folder_id")
        GDRIVE_CREATE_YM_FOLDERS = bot_config.get("gdrive_create_ym_folders", DEFAULT_CONFIG["gdrive_create_ym_folders"])
        new_gdrive_key_path = bot_config.get("gdrive_service_account_key_path", DEFAULT_CONFIG["gdrive_service_account_key_path"])
        path_changed = (GDRIVE_SERVICE_ACCOUNT_KEY_PATH != new_gdrive_key_path)
        GDRIVE_SERVICE_ACCOUNT_KEY_PATH = new_gdrive_key_path
        if path_changed or ("gdrive_target_folder_id" in new_settings and new_settings["gdrive_target_folder_id"]):
             print("Google Drive関連の設定が変更されたため、サービスを再初期化します。")
             initialize_gdrive_service()
    except Exception as e: print(f"エラー: '{CONFIG_FILE_NAME}' の保存中に問題が発生しました: {e}")

def extract_gdrive_folder_id_from_string(input_string: str) -> str:
    match = re.search(r"folders/([a-zA-Z0-9_-]{25,})", input_string)
    if match:
        extracted_id = match.group(1)
        print(f"URLからGoogle DriveフォルダIDを抽出しました: {extracted_id}")
        return extracted_id
    match_id_param = re.search(r"[?&]id=([a-zA-Z0-9_-]{25,})", input_string)
    if match_id_param:
        extracted_id = match_id_param.group(1)
        print(f"URLパラメータからGoogle DriveフォルダIDを抽出しました: {extracted_id}")
        return extracted_id
    print(f"入力文字列をそのままGoogle DriveフォルダIDとして扱います: {input_string}")
    return input_string.strip()

def is_admin():
    async def predicate(interaction: discord.Interaction):
        if interaction.guild is None:
            await interaction.response.send_message("このコマンドはサーバー内でのみ実行可能です。", ephemeral=True); return False
        if not ADMIN_ROLE_NAMES:
             await interaction.response.send_message("実行に必要なロールがBOTに設定されていません。BOT管理者にお問い合わせください。", ephemeral=True); return False
        author_roles = [role.name for role in interaction.user.roles]
        if any(admin_role in author_roles for admin_role in ADMIN_ROLE_NAMES): return True
        await interaction.response.send_message(f"このコマンドの実行には、次のいずれかのロールが必要です: `{', '.join(ADMIN_ROLE_NAMES)}`", ephemeral=True); return False
    return app_commands.check(predicate)

async def get_tags_from_gemini(file_path, original_filename, mime_type):
    global gemini_model_instance
    if not gemini_model_instance: print("Geminiモデルが初期化されていないため、タグ生成をスキップします。"); return "notags"
    print(f"Gemini APIにファイル '{original_filename}' (MIMEタイプ: {mime_type}) を送信してタグを生成します...")
    uploaded_file_resource = None
    try:
        uploaded_file_resource = genai.upload_file(path=file_path, display_name=original_filename)
        print(f"Gemini APIにファイル '{original_filename}' (ID: {uploaded_file_resource.name}) をアップロードしました。")
        prompt = load_tagging_prompt()
        response = await gemini_model_instance.generate_content_async([prompt, uploaded_file_resource], generation_config={"response_mime_type": "text/plain"})
        if response.text.strip() == "タグ抽出不可": print("Gemini API: タグ抽出不可と判断されました。"); return "notags"
        tags = response.text.strip()
        sanitized_tags = sanitize_filename_component(tags)
        print(f"Gemini APIから取得したタグ: '{sanitized_tags}'")
        return sanitized_tags if sanitized_tags else "notags"
    except Exception as e: print(f"Gemini APIでのタグ生成中にエラーが発生しました: {e}"); return "notags"
    finally:
        if uploaded_file_resource and hasattr(uploaded_file_resource, 'name'):
             try:
                 print(f"Gemini APIからアップロードされたファイル '{uploaded_file_resource.name}' の削除を試みます...")
                 genai.delete_file(uploaded_file_resource.name)
                 print(f"Gemini APIからアップロードされたファイル '{uploaded_file_resource.name}' を削除しました。")
             except Exception as e_del: print(f"Gemini APIからアップロードされたファイル {uploaded_file_resource.name} の削除中にエラー: {e_del}")

def get_or_create_drive_folder(parent_folder_id: str, folder_name: str) -> str | None:
    if not gdrive_service or not google_drive_libs_available:
        print("Driveサービスが利用不可のため、フォルダ操作はできません。"); return None
    try:
        query = f"mimeType='application/vnd.google-apps.folder' and trashed=false and name='{folder_name}' and '{parent_folder_id}' in parents"
        response = gdrive_service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        folders = response.get('files', [])
        if folders:
            print(f"Driveフォルダ '{folder_name}' が見つかりました (ID: {folders[0].get('id')})。"); return folders[0].get('id')
        else:
            print(f"Driveフォルダ '{folder_name}' が見つからないため、作成します...")
            file_metadata = {'name': folder_name, 'mimeType': 'application/vnd.google-apps.folder', 'parents': [parent_folder_id]}
            folder = gdrive_service.files().create(body=file_metadata, fields='id').execute()
            print(f"Driveフォルダ '{folder_name}' を作成しました (ID: {folder.get('id')})。"); return folder.get('id')
    except Exception as e: print(f"Driveフォルダ '{folder_name}' の検索または作成中にエラー: {e}"); return None

async def upload_to_gdrive(local_file_path: str, drive_filename: str, attachment_content_type: str) -> dict | None:
    if not gdrive_service or not google_drive_libs_available:
        print("Google Driveサービスが利用できないため、アップロードをスキップします。"); return None
    if not GDRIVE_TARGET_FOLDER_ID:
        print("Google DriveのターゲットフォルダIDが設定されていません。アップロードをスキップします。"); return None
    parent_id_to_upload = GDRIVE_TARGET_FOLDER_ID
    if GDRIVE_CREATE_YM_FOLDERS:
        now = datetime.datetime.now()
        year_month_folder_name = now.strftime("%Y%m")
        ym_drive_folder_id = get_or_create_drive_folder(GDRIVE_TARGET_FOLDER_ID, year_month_folder_name)
        if ym_drive_folder_id: parent_id_to_upload = ym_drive_folder_id
        else: print(f"年月フォルダ '{year_month_folder_name}' の準備に失敗したため、設定されたメインターゲットフォルダにアップロードします。")
    file_metadata = {'name': drive_filename, 'parents': [parent_id_to_upload]}
    try:
        mime_type = attachment_content_type if attachment_content_type else 'application/octet-stream'
        media = MediaFileUpload(local_file_path, mimetype=mime_type, resumable=True)
        print(f"Google Drive ({parent_id_to_upload}) へ '{drive_filename}' をアップロード開始...")
        uploaded_file = gdrive_service.files().create(body=file_metadata, media_body=media, fields='id, name, webViewLink, thumbnailLink, size').execute()
        print(f"ファイル '{uploaded_file.get('name')}' がGoogle Driveにアップロードされました。ID: {uploaded_file.get('id')}, Link: {uploaded_file.get('webViewLink')}")
        return uploaded_file
    except Exception as e: print(f"Google Driveへのファイルアップロード中にエラーが発生しました: {e}"); return None

class ConfirmDeleteView(discord.ui.View):
    def __init__(self, author_id: int, file_path_to_delete: str, filename_display: str):
        super().__init__(timeout=30.0); self.author_id = author_id; self.file_path_to_delete = file_path_to_delete
        self.filename_display = filename_display; self.confirmed: bool | None = None; self.interaction_message: discord.InteractionMessage | None = None
    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.author_id:
            await interaction.response.send_message("この操作はコマンドを実行した本人のみが行えます。", ephemeral=True); return False
        return True
    @discord.ui.button(label="削除実行", style=discord.ButtonStyle.danger)
    async def confirm_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.confirmed = True; [setattr(item, 'disabled', True) for item in self.children]
        await interaction.response.edit_message(content=f"ファイル `{self.filename_display}` の削除処理を開始します...", view=self); self.stop()
    @discord.ui.button(label="キャンセル", style=discord.ButtonStyle.secondary)
    async def cancel_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.confirmed = False; [setattr(item, 'disabled', True) for item in self.children]
        await interaction.response.edit_message(content=f"ファイル `{self.filename_display}` の削除はキャンセルされました。", view=self); self.stop()
    async def on_timeout(self):
        if self.confirmed is None:
            [setattr(item, 'disabled', True) for item in self.children]
            if self.interaction_message:
                try: await self.interaction_message.edit(content=f"ファイル `{self.filename_display}` の削除確認がタイムアウトしました。", view=self)
                except discord.NotFound: pass
                except discord.HTTPException as e: print(f"タイムアウト時のメッセージ編集エラー: {e}")
            self.stop()

@bot.event
async def on_ready():
    global current_gemini_model
    print(f'{bot.user.name} としてログインしました (ID: {bot.user.id})')
    print(f'監視中のサーバー数: {len(bot.guilds)}')
    print(f'ベースアップロードフォルダ(ローカル): {os.path.abspath(BASE_UPLOAD_FOLDER)}')
    print(f'現在のアップロード先: {UPLOAD_DESTINATION}')
    print(f'Geminiコマンド管理者ロール: {ADMIN_ROLE_NAMES}')
    if gemini_model_instance: print(f'使用中Geminiモデル: {current_gemini_model}')
    else: print('Geminiモデルは初期化されていません。')
    load_tagging_prompt()
    if not os.path.exists(BASE_UPLOAD_FOLDER) and UPLOAD_DESTINATION == "local":
        os.makedirs(BASE_UPLOAD_FOLDER); print(f"ベースフォルダ '{BASE_UPLOAD_FOLDER}' を作成しました。")
    initialize_gdrive_service()
    try: await bot.tree.sync(); print("スラッシュコマンドを同期しました。")
    except Exception as e: print(f"スラッシュコマンドの同期に失敗しました: {e}")
    print('------')

@bot.event
async def on_message(message):
    if message.author == bot.user: return
    if message.attachments:
        ctx = await bot.get_context(message)
        for attachment in message.attachments:
            allowed_image_types = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp')
            allowed_video_types = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
            file_ext = os.path.splitext(attachment.filename)[1].lower()
            if not (file_ext in allowed_image_types or file_ext in allowed_video_types):
                await message.channel.send(f"ファイル '{attachment.filename}' の形式 ({file_ext}) はサポートされていません。"); continue
            limit_bytes = 8 * 1024 * 1024
            if ctx.guild: limit_bytes = ctx.guild.filesize_limit
            if attachment.size > limit_bytes:
                 await message.channel.send(f"ファイル '{attachment.filename}' ({attachment.size // 1024 // 1024}MB) はサイズが大きすぎます (上限: {limit_bytes // 1024 // 1024}MB)。"); continue
            temp_dir = os.path.join(BASE_UPLOAD_FOLDER, "temp")
            if not os.path.exists(temp_dir): os.makedirs(temp_dir)
            temp_save_path = os.path.join(temp_dir, f"temp_{attachment.id}_{sanitize_filename_component(attachment.filename)}")
            await attachment.save(temp_save_path)
            processing_msg = await message.channel.send(f"ファイル '{attachment.filename}' を処理中... 自動タグ付けを開始します。")
            tags_str = "notags"
            if gemini_model_instance:
                try:
                    if file_ext in allowed_image_types:
                        try: img = Image.open(temp_save_path); img.verify(); img.close()
                        except Exception as img_err:
                            await processing_msg.edit(content=f"ファイル '{attachment.filename}' は有効な画像ではないようです。処理中断。({img_err})")
                            if os.path.exists(temp_save_path): os.remove(temp_save_path); continue
                    tags_str = await get_tags_from_gemini(temp_save_path, attachment.filename, attachment.content_type)
                except Exception as e:
                    print(f"タグ付け処理中にエラー: {e}")
                    await processing_msg.edit(content=f"ファイル '{attachment.filename}' のタグ付け中にエラー。タグなしで処理します。"); tags_str = "notags"
            else: await processing_msg.edit(content=f"ファイル '{attachment.filename}' を処理中... (Gemini API未設定のためタグ付けスキップ)")
            date_str = datetime.datetime.now().strftime("%Y%m%d")
            original_filename_no_ext, original_ext = os.path.splitext(attachment.filename)
            sanitized_original_filename = sanitize_filename_component(original_filename_no_ext)
            new_filename = f"{date_str}_{tags_str}_{sanitized_original_filename}{original_ext}"
            display_tags_on_message = tags_str.replace("_", "-") if tags_str != "notags" else "なし"

            current_upload_dest = bot_config.get("upload_destination", "local")
            if current_upload_dest == "gdrive":
                if gdrive_service and GDRIVE_TARGET_FOLDER_ID:
                    gdrive_file_info = await upload_to_gdrive(temp_save_path, new_filename, attachment.content_type)
                    if gdrive_file_info:
                        file_link = gdrive_file_info.get('webViewLink', 'リンク不明')
                        await processing_msg.edit(content=(f"ファイル '{attachment.filename}' をGoogle Driveにアップロードし、'{new_filename}' として保存しました。\n"
                                                           f"自動タグ: `{display_tags_on_message}`\nリンク: <{file_link}>"))
                    else: await processing_msg.edit(content=f"ファイル '{attachment.filename}' のGoogle Driveへのアップロードに失敗しました。")
                else: await processing_msg.edit(content=f"Google Driveが設定されていないため、 '{attachment.filename}' のアップロードをスキップ。")
                if os.path.exists(temp_save_path):
                    try: os.remove(temp_save_path); print(f"一時ファイル '{temp_save_path}' を削除しました。")
                    except Exception as e_rm: print(f"一時ファイル '{temp_save_path}' の削除失敗: {e_rm}")
            elif current_upload_dest == "local":
                local_ym_folder = create_year_month_folder_if_not_exists(BASE_UPLOAD_FOLDER)
                final_save_path = os.path.join(local_ym_folder, new_filename)
                try:
                    os.rename(temp_save_path, final_save_path)
                    print(f"ファイル '{attachment.filename}' を '{final_save_path}' に保存しました。")
                    await processing_msg.edit(content=(f"ファイル '{attachment.filename}' をローカルに保存しました: '{new_filename}'\n自動タグ: `{display_tags_on_message}`"))
                except Exception as e:
                    print(f"ローカル保存エラー: {e}"); await processing_msg.edit(content=f"'{attachment.filename}' のローカル保存中にエラー。")
                    if os.path.exists(temp_save_path):
                        try: os.remove(temp_save_path); print(f"一時ファイル '{temp_save_path}' を削除しました。")
                        except Exception as e_rm: print(f"一時ファイル '{temp_save_path}' の削除失敗: {e_rm}")
            else: print(f"不明なアップロード先: {current_upload_dest}")
    await bot.process_commands(message)

# --- オートコンプリート用の関数 ---
async def gemini_model_autocomplete(interaction: discord.Interaction, current: str) -> list[app_commands.Choice[str]]:
    choices = []
    if not GEMINI_API_KEY or not genai: return []
    try:
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                model_display_name = model.name.replace("models/", "")
                if current.lower() in model_display_name.lower():
                    choice_name = f"{model_display_name} ({model.display_name})"
                    if len(choice_name) > 100: choice_name = model_display_name[:97] + "..."
                    choices.append(app_commands.Choice(name=choice_name, value=model_display_name))
            if len(choices) >= 25: break
    except Exception as e: print(f"Geminiモデルのオートコンプリート中にエラー: {e}"); return []
    return choices
async def year_month_autocomplete(interaction: discord.Interaction, current: str) -> list[app_commands.Choice[str]]:
    choices = []; ym_folders = set()
    if not os.path.exists(BASE_UPLOAD_FOLDER): return []
    try:
        for item in os.listdir(BASE_UPLOAD_FOLDER):
            if os.path.isdir(os.path.join(BASE_UPLOAD_FOLDER, item)) and len(item) == 6 and item.isdigit(): ym_folders.add(item)
        for folder_name in sorted(list(ym_folders), reverse=True):
            if current.lower() in folder_name.lower(): choices.append(app_commands.Choice(name=folder_name, value=folder_name))
            if len(choices) >= 25: break
    except Exception as e: print(f"year_month_autocomplete 中にエラー: {e}"); return []
    return choices
async def filename_autocomplete(interaction: discord.Interaction, current: str) -> list[app_commands.Choice[str]]:
    choices = []; specific_ym_folder = None; current_filename_part_to_search = current
    if not os.path.exists(BASE_UPLOAD_FOLDER): return []
    if '/' in current and len(current.split('/')[0]) == 6 and current.split('/')[0].isdigit():
        parts = current.split('/', 1); ym_part = parts[0]
        path_to_check = os.path.join(BASE_UPLOAD_FOLDER, ym_part)
        if os.path.isdir(path_to_check): specific_ym_folder = path_to_check; current_filename_part_to_search = parts[1] if len(parts) > 1 else ""
    folders_to_search = [specific_ym_folder] if specific_ym_folder else \
                        [os.path.join(BASE_UPLOAD_FOLDER, item) for item in sorted(os.listdir(BASE_UPLOAD_FOLDER), reverse=True)
                         if os.path.isdir(os.path.join(BASE_UPLOAD_FOLDER, item)) and len(item) == 6 and item.isdigit()]
    for folder_path in folders_to_search:
        year_month_dir = os.path.basename(folder_path)
        try:
            for fname in sorted(os.listdir(folder_path)):
                if os.path.isfile(os.path.join(folder_path, fname)):
                    if current_filename_part_to_search.lower() in fname.lower():
                        suffix = f" (in {year_month_dir})"; allowed_fname_len = 100 - len(suffix)
                        if allowed_fname_len < 1: display_fname = ""
                        elif len(fname) <= allowed_fname_len: display_fname = fname
                        else: display_fname = fname[:max(0, allowed_fname_len - 3)] + "..."
                        final_choice_name = f"{display_fname}{suffix}"
                        if len(final_choice_name) > 100: final_choice_name = final_choice_name[:97] + "..."
                        choices.append(app_commands.Choice(name=final_choice_name, value=f"{year_month_dir}/{fname}"))
                if len(choices) >= 25: break
        except Exception as e: print(f"filename_autocomplete でフォルダ '{folder_path}' のスキャン中にエラー: {e}")
        if len(choices) >= 25: break
    return choices

# --- コマンドグループの定義 ---
gemini_group = app_commands.Group(name="gemini", description="Geminiモデル関連の操作を行います。")
files_group = app_commands.Group(name="files", description="アップロードされたファイルの管理を行います。")
upload_settings_group = app_commands.Group(name="upload_settings", description="ファイルのアップロード先設定を管理します。")

# --- スラッシュコマンド ---
@bot.tree.command(name="upload_guide", description="ファイルアップロード方法の案内")
async def upload_guide(interaction: discord.Interaction):
    await interaction.response.send_message("ファイルをアップロードするには、このチャンネルに直接ファイルをドラッグ＆ドロップするか、メッセージ入力欄の「+」ボタンからファイルを添付して送信してください。\n画像または動画ファイルが対象です。")

# --- /files サブコマンド ---
@files_group.command(name="list", description="保存されているファイルの一覧を表示します。")
@app_commands.describe(year_month="表示する年月 (例: 202305)。", keyword="ファイル名やタグに含まれるキーワードで絞り込みます。")
@app_commands.autocomplete(year_month=year_month_autocomplete)
async def files_list(interaction: discord.Interaction, year_month: str = None, keyword: str = None):
    await interaction.response.defer()
    found_files_details = []; search_paths = []
    if year_month:
        if not (len(year_month) == 6 and year_month.isdigit()):
            await interaction.followup.send("年月の指定が正しくありません。YYYYMM形式で入力してください (例: 202305)。"); return
        target_ym_folder = os.path.join(BASE_UPLOAD_FOLDER, year_month)
        if os.path.exists(target_ym_folder) and os.path.isdir(target_ym_folder): search_paths.append(target_ym_folder)
        else: await interaction.followup.send(f"指定された年月フォルダ '{year_month}' は見つかりません。"); return
    else:
        if os.path.exists(BASE_UPLOAD_FOLDER):
            for item in sorted(os.listdir(BASE_UPLOAD_FOLDER), reverse=True):
                item_path = os.path.join(BASE_UPLOAD_FOLDER, item)
                if os.path.isdir(item_path) and len(item) == 6 and item.isdigit(): search_paths.append(item_path)
    if not search_paths:
        msg = "検索対象のフォルダが見つかりません。"
        if year_month: msg = f"指定された年月フォルダ '{year_month}' は見つかりません。"
        elif not os.path.exists(BASE_UPLOAD_FOLDER): msg = f"ベースアップロードフォルダ '{BASE_UPLOAD_FOLDER}' が見つかりません。"
        await interaction.followup.send(msg); return
    for folder_to_scan in search_paths:
        try:
            for fname in sorted(os.listdir(folder_to_scan)):
                if os.path.isfile(os.path.join(folder_to_scan, fname)):
                    if keyword and keyword.lower() not in fname.lower(): continue
                    parsed_info = parse_bot_filename(fname)
                    found_files_details.append({
                        "fullname": fname, "date": parsed_info["date"], "tags": parsed_info["tags_display"],
                        "original_name": parsed_info["original_stem"], "year_month": os.path.basename(folder_to_scan)
                    })
        except Exception as e: print(f"フォルダ '{folder_to_scan}' のスキャン中にエラー: {e}"); await interaction.followup.send(f"フォルダ '{os.path.basename(folder_to_scan)}' の一覧取得中にエラーが発生しました。"); return
    if not found_files_details:
        message = "ファイルは見つかりませんでした。";
        if year_month: message += f" (年月: {year_month})"
        if keyword: message += f" (キーワード: {keyword})"
        await interaction.followup.send(message); return
    embed = discord.Embed(title="ファイル一覧", color=discord.Color.blue())
    description_parts = []
    if year_month: description_parts.append(f"年月: `{year_month}`")
    if keyword: description_parts.append(f"キーワード: `{keyword}`")
    if description_parts: embed.description = "絞り込み条件: " + " | ".join(description_parts)
    MAX_FILES_IN_EMBED = 10
    for i, file_info in enumerate(found_files_details):
        if i >= MAX_FILES_IN_EMBED:
            embed.add_field(name="...", value=f"他 {len(found_files_details) - MAX_FILES_IN_EMBED} 件のファイルがあります。", inline=False); break
        field_name = f"📁 `{file_info['fullname']}`"
        field_value = (f"元ファイル名: `{file_info['original_name']}`\nタグ: `{file_info['tags']}`\n保存日: `{file_info['date']}` (in `{file_info['year_month']}`)")
        embed.add_field(name=field_name, value=field_value, inline=False)
    if not embed.fields: await interaction.followup.send("表示できるファイル情報がありません。"); return
    await interaction.followup.send(embed=embed)

@files_group.command(name="info", description="指定された保存済みファイルの詳細情報を表示します。")
@app_commands.describe(filepath="情報を表示するファイル (年月フォルダ/ファイル名)")
@app_commands.autocomplete(filepath=filename_autocomplete)
async def files_info(interaction: discord.Interaction, filepath: str):
    await interaction.response.defer()
    try: ym_dir, filename = filepath.split('/', 1)
    except ValueError: await interaction.followup.send("ファイルパスの形式が正しくありません。", ephemeral=True); return
    full_path = os.path.join(BASE_UPLOAD_FOLDER, ym_dir, filename)
    if not os.path.exists(full_path) or not os.path.isfile(full_path):
        await interaction.followup.send(f"ファイル `{filepath}` が見つかりません。"); return
    try:
        parsed_info = parse_bot_filename(filename); file_size_bytes = os.path.getsize(full_path)
        file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
        embed = discord.Embed(title=f"ファイル情報: {filename}", color=discord.Color.green())
        embed.add_field(name="フルパス (サーバー上)", value=f"`{full_path}`", inline=False)
        embed.add_field(name="年月フォルダ", value=f"`{ym_dir}`", inline=True)
        embed.add_field(name="ファイルサイズ", value=f"{file_size_bytes} Bytes ({file_size_mb} MB)", inline=True)
        embed.add_field(name="元ファイル名 (拡張子除く)", value=f"`{parsed_info['original_stem']}`", inline=False)
        embed.add_field(name="拡張子", value=f"`{parsed_info['extension']}`", inline=True)
        embed.add_field(name="抽出されたタグ", value=f"`{parsed_info['tags_display']}`", inline=True)
        embed.add_field(name="抽出された日付", value=f"`{parsed_info['date']}`", inline=True)
        try:
            m_time = os.path.getmtime(full_path)
            modified_time = datetime.datetime.fromtimestamp(m_time).strftime('%Y-%m-%d %H:%M:%S')
            embed.add_field(name="最終更新日時 (サーバー)", value=modified_time, inline=False)
        except Exception as e_time: print(f"最終更新日時の取得エラー: {e_time}")
        await interaction.followup.send(embed=embed)
    except Exception as e: print(f"/files info処理中にエラー: {e}"); await interaction.followup.send(f"ファイル情報の取得中にエラー: {e}")

@files_group.command(name="delete", description="指定された保存済みファイルをサーバーから削除します。")
@app_commands.describe(filepath="削除するファイル (年月フォルダ/ファイル名)")
@app_commands.autocomplete(filepath=filename_autocomplete)
async def files_delete(interaction: discord.Interaction, filepath: str):
    await interaction.response.defer()
    try: ym_dir, filename = filepath.split('/', 1)
    except ValueError: await interaction.followup.send("ファイルパスの形式が正しくありません。", ephemeral=True); return
    full_path = os.path.join(BASE_UPLOAD_FOLDER, ym_dir, filename)
    if not os.path.exists(full_path) or not os.path.isfile(full_path):
        await interaction.followup.send(f"ファイル `{filepath}` が見つかりません。"); return
    view = ConfirmDeleteView(author_id=interaction.user.id, file_path_to_delete=full_path, filename_display=filename)
    interaction_message = await interaction.followup.send(f"**警告:** ファイル `{filename}` を本当に削除しますか？この操作は取り消せません。(実行者: {interaction.user.mention})", view=view)
    view.interaction_message = interaction_message
    await view.wait()
    if view.confirmed is True:
        try:
            os.remove(full_path); print(f"ユーザー {interaction.user} によってファイル {full_path} が削除されました。")
            await interaction_message.edit(content=f"ファイル `{filename}` を削除しました。(実行者: {interaction.user.mention})", view=None)
        except Exception as e: print(f"ファイル削除エラー ({full_path}): {e}"); await interaction_message.edit(content=f"ファイル `{filename}` の削除中にエラー: {e}", view=None)

@files_group.command(name="get", description="指定された保存済みファイルを取得します。")
@app_commands.describe(filepath="取得するファイル (年月フォルダ/ファイル名)")
@app_commands.autocomplete(filepath=filename_autocomplete)
async def files_get(interaction: discord.Interaction, filepath: str):
    await interaction.response.defer()
    try: ym_dir, filename = filepath.split('/', 1)
    except ValueError: await interaction.followup.send("ファイルパスの形式が正しくありません。", ephemeral=True); return
    full_path = os.path.join(BASE_UPLOAD_FOLDER, ym_dir, filename)
    if not os.path.exists(full_path) or not os.path.isfile(full_path):
        await interaction.followup.send(f"ファイル `{filepath}` が見つかりません。"); return
    limit_bytes = 8 * 1024 * 1024
    if interaction.guild: limit_bytes = interaction.guild.filesize_limit
    file_size_bytes = os.path.getsize(full_path)
    if file_size_bytes > limit_bytes:
        await interaction.followup.send(f"ファイル `{filename}` ({round(file_size_bytes / (1024*1024), 2)} MB) はサイズが大きすぎます (上限: {round(limit_bytes / (1024*1024), 2)} MB)"); return
    try:
        discord_file = discord.File(full_path, filename=filename)
        await interaction.followup.send(f"ファイル `{filename}` を送信します: (要求者: {interaction.user.mention})", file=discord_file)
    except Exception as e: print(f"ファイル送信エラー ({full_path}): {e}"); await interaction.followup.send(f"ファイル `{filename}` の送信中にエラー: {e}")

# --- /gemini サブコマンド ---
@gemini_group.command(name="list", description="利用可能なGeminiモデルの一覧を表示します。(ロール制限あり)")
@is_admin()
async def gemini_list(interaction: discord.Interaction):
    if not GEMINI_API_KEY: await interaction.response.send_message("Gemini APIキーが設定されていません。", ephemeral=True); return
    if not genai: await interaction.response.send_message("Geminiライブラリが利用できません。", ephemeral=True); return
    await interaction.response.defer(ephemeral=True)
    try:
        models_info_parts = ["利用可能なGeminiモデル (generateContentサポート):\n"]; count = 0
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                model_display_name = model.name.replace("models/", ""); current_part = f"- `{model_display_name}` ({model.display_name})\n"
                if len("".join(models_info_parts)) + len(current_part) > 1900:
                    await interaction.followup.send("".join(models_info_parts), ephemeral=True); models_info_parts = [current_part]
                else: models_info_parts.append(current_part)
                count += 1
        if count == 0 and len(models_info_parts) == 1 and models_info_parts[0].endswith(":\n"): models_info_parts.append("利用可能なGeminiモデルが見つかりませんでした。")
        if models_info_parts:
            final_message = "".join(models_info_parts)
            if final_message.strip() and not (count == 0 and final_message.endswith(":\n") and len(final_message.splitlines()) ==1):
                 await interaction.followup.send(final_message, ephemeral=True)
            elif count == 0 : await interaction.followup.send("利用可能なGeminiモデル (generateContentサポート) が見つかりませんでした。",ephemeral=True)
    except Exception as e: await interaction.followup.send(f"モデル一覧の取得中にエラー: {e}", ephemeral=True)

@gemini_group.command(name="set", description="自動タグ付けに使用するGeminiモデルを設定します。(ロール制限あり)")
@app_commands.describe(model_name="Geminiモデル名 (例: gemini-1.5-flash-latest)。")
@app_commands.autocomplete(model_name=gemini_model_autocomplete)
@is_admin()
async def gemini_set(interaction: discord.Interaction, model_name: str):
    global current_gemini_model, gemini_model_instance
    if not GEMINI_API_KEY: await interaction.response.send_message("Gemini APIキーが設定されていません。", ephemeral=True); return
    if not genai: await interaction.response.send_message("Geminiライブラリが利用できません。", ephemeral=True); return
    await interaction.response.defer(ephemeral=True)
    try:
        full_model_name_to_check = model_name if model_name.startswith("models/") else f"models/{model_name}"
        retrieved_model = genai.get_model(full_model_name_to_check)
        if 'generateContent' not in retrieved_model.supported_generation_methods:
            await interaction.followup.send(f"モデル `{model_name}` は `generateContent` をサポートしていません。", ephemeral=True); return
        new_model_instance = genai.GenerativeModel(retrieved_model.name, safety_settings={ HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,})
        current_gemini_model = retrieved_model.name.replace("models/", "")
        gemini_model_instance = new_model_instance
        await interaction.followup.send(f"自動タグ付けのGeminiモデルを `{current_gemini_model}` に設定しました。", ephemeral=True)
        print(f"Geminiモデルが '{current_gemini_model}' に変更されました。 (実行者: {interaction.user})")
    except Exception as e: await interaction.followup.send(f"モデル `{model_name}` の設定に失敗: {e}", ephemeral=True); print(f"Geminiモデル '{model_name}' の設定失敗: {e}")

@gemini_group.command(name="current", description="現在設定されているGeminiモデル名を表示します。(ロール制限あり)")
@is_admin()
async def gemini_current(interaction: discord.Interaction):
    if not gemini_model_instance: await interaction.response.send_message(f"Geminiモデルは現在設定されていません、または初期化に失敗しています。", ephemeral=True)
    else: await interaction.response.send_message(f"現在設定されているGeminiモデルは `{current_gemini_model}` です。", ephemeral=True)

# --- /upload_settings コマンド ---
@upload_settings_group.command(name="set_destination", description="ファイルのアップロード先を設定します。(ロール制限あり)")
@app_commands.describe(destination="アップロード先 ('local' または 'gdrive')")
@app_commands.choices(destination=[app_commands.Choice(name="ローカルストレージ", value="local"), app_commands.Choice(name="Google Drive", value="gdrive"),])
@is_admin()
async def set_upload_destination(interaction: discord.Interaction, destination: app_commands.Choice[str]):
    await interaction.response.defer(ephemeral=True)
    new_destination_value = destination.value
    if new_destination_value not in ["local", "gdrive"]:
        await interaction.followup.send("無効なアップロード先です。'local' または 'gdrive' を指定してください。", ephemeral=True); return
    if new_destination_value == "gdrive":
        if not GDRIVE_TARGET_FOLDER_ID:
            await interaction.followup.send("Google Drive をアップロード先に設定する前に、`/upload_settings set_gdrive_folder` コマンドでターゲットフォルダIDを設定してください。", ephemeral=True); return
        if not gdrive_service and google_drive_libs_available: initialize_gdrive_service()
        if not gdrive_service:
            await interaction.followup.send("Google Driveサービスが利用できません。設定を確認してください。", ephemeral=True); return
    save_bot_config({"upload_destination": new_destination_value})
    await interaction.followup.send(f"ファイルのアップロード先を「{destination.name}」に設定しました。", ephemeral=True)
    print(f"アップロード先が '{new_destination_value}' に変更されました。(実行者: {interaction.user})")

@upload_settings_group.command(name="set_gdrive_folder", description="Google Driveのアップロード先フォルダIDまたはURLを設定します。(ロール制限あり)")
@app_commands.describe(folder_id_or_url="Google DriveのフォルダID、またはフォルダのURL")
@is_admin()
async def set_gdrive_folder_id(interaction: discord.Interaction, folder_id_or_url: str):
    await interaction.response.defer(ephemeral=True)
    extracted_folder_id = extract_gdrive_folder_id_from_string(folder_id_or_url)
    if not extracted_folder_id or len(extracted_folder_id) < 20:
        await interaction.followup.send(f"設定しようとしているフォルダID「{extracted_folder_id}」は無効な形式のようです。\n正しいGoogle DriveのフォルダID（20文字以上の英数字とハイフン/アンダースコア）、またはフォルダURLを入力してください。", ephemeral=True); return
    save_bot_config({"gdrive_target_folder_id": extracted_folder_id})
    await interaction.followup.send(f"Google Driveのアップロード先フォルダIDを `{extracted_folder_id}` に設定しました。\n(入力値: `{folder_id_or_url}`)", ephemeral=True)
    print(f"GdriveターゲットフォルダIDが '{extracted_folder_id}' に変更されました。(実行者: {interaction.user})")

@upload_settings_group.command(name="current_settings", description="現在のファイルアップロード設定を表示します。(ロール制限あり)")
@is_admin()
async def current_upload_settings(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    dest = bot_config.get("upload_destination", DEFAULT_CONFIG["upload_destination"])
    folder_id = bot_config.get("gdrive_target_folder_id", "未設定")
    create_ym = bot_config.get("gdrive_create_ym_folders", DEFAULT_CONFIG["gdrive_create_ym_folders"])
    gdrive_key_path = bot_config.get("gdrive_service_account_key_path", "未設定")
    embed = discord.Embed(title="現在のアップロード設定", color=discord.Color.blue())
    embed.add_field(name="アップロード先", value=f"`{dest}`", inline=False)
    embed.add_field(name="Google Drive フォルダID", value=f"`{folder_id}`", inline=False)
    embed.add_field(name="Google Drive 年月フォルダ作成", value=f"`{create_ym}`", inline=False)
    embed.add_field(name="Google Drive サービスキーパス", value=f"`{gdrive_key_path}`", inline=False)
    gdrive_status = "初期化成功" if gdrive_service else ("未初期化または失敗" if google_drive_libs_available else "ライブラリ不足")
    embed.add_field(name="Google Drive サービス状態", value=gdrive_status, inline=False)
    await interaction.followup.send(embed=embed, ephemeral=True)

@bot.tree.command(name="help_nasbot", description="このBOTのコマンド一覧と簡単な説明を表示します。")
async def help_nasbot(interaction: discord.Interaction):
    embed = discord.Embed(title="ファイル管理BOT ヘルプ", description="このBOTで利用可能なコマンド一覧です。", color=discord.Color.blue())
    embed.add_field(name="ファイル管理 (`/files`)", value=("`  list [year_month] [keyword]` - 保存されたファイルの一覧を表示します。\n"
                                                       "`  info <filepath>` - 指定されたファイルの詳細情報を表示します。\n"
                                                       "`  get <filepath>` - 指定されたファイルを取得します。\n"
                                                       "`  delete <filepath>` - 指定されたファイルを削除します。\n"), inline=False)
    embed.add_field(name="アップロード設定 (`/upload_settings`) (指定ロールのみ)", value=(
        "`  set_destination <local|gdrive>` - アップロード先を設定します。\n"
        "`  set_gdrive_folder <folder_id_or_url>` - Google Driveの保存先フォルダID/URLを設定します。\n" # 説明更新
        "`  current_settings` - 現在のアップロード関連設定を表示します。\n"), inline=False)
    embed.add_field(name="Geminiモデル設定 (`/gemini`) (指定ロールのみ)", value=(
        "`  set <model_name>` - 自動タグ付けに使用するGeminiモデルを設定します。\n"
        "`  current` - 現在のGeminiモデル名を表示します。\n"
        "`  list` - 利用可能なGeminiモデルの一覧を表示します。\n"), inline=False)
    embed.add_field(name="その他", value=("`/upload_guide` - ファイルのアップロード方法を表示します。\n"
                                      "`/help_nasbot` - このヘルプを表示します。"), inline=False)
    embed.set_footer(text="ファイルを直接このチャンネルにアップロードすることでも処理が開始されます。")
    await interaction.response.send_message(embed=embed)

# --- コマンドグループをBOTに追加 ---
bot.tree.add_command(gemini_group)
bot.tree.add_command(files_group)
bot.tree.add_command(upload_settings_group)

# --- BOT実行 ---
if __name__ == "__main__":
    if DISCORD_BOT_TOKEN:
        if not GEMINI_API_KEY: print("警告: GEMINI_API_KEYが .envファイルに設定されていません。")
        if not google_drive_libs_available:
            print("警告: Google Drive連携に必要なライブラリが不足しているため、Google Drive関連機能は動作しません。")
        bot.run(DISCORD_BOT_TOKEN)
    else: print("エラー: DISCORD_BOT_TOKEN が .envファイルに設定されていません。BOTを起動できません。")