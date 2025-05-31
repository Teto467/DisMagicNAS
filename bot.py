import discord
from discord.ext import commands
import os
import datetime
import re
import asyncio # asyncio をインポート
import json
import asyncio
import sys
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from PIL import Image
from discord import app_commands
import io # GDriveからダウンロードする際に使用


if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
# Google Drive API 関連のインポート
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload # MediaIoBaseDownload を追加
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
intents.members = True # メンバーインテントの追加
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
    match = re.match(r"(\d{8})_([^_]+)_(.+)", base_name) # タグ部分はアンダースコアを含まない前提
    if match:
        parts["date"], parts["tags_raw"], parts["original_stem"] = match.groups()
        if parts["tags_raw"] == "notags": parts["tags_display"] = "タグなし"
        else: parts["tags_display"] = parts["tags_raw"].replace("_", "-") # 表示用にアンダースコアをハイフンに
    else:
        match_no_tags = re.match(r"(\d{8})_(.+)", base_name)
        if match_no_tags: parts["date"], parts["original_stem"] = match_no_tags.groups()
        else: parts["original_stem"] = base_name # 日付もタグもない場合は全体を元のファイル名とする
    return parts

def save_bot_config(new_settings: dict):
    global bot_config, UPLOAD_DESTINATION, GDRIVE_TARGET_FOLDER_ID, GDRIVE_CREATE_YM_FOLDERS, GDRIVE_SERVICE_ACCOUNT_KEY_PATH, gdrive_service
    current_full_config = {}
    if os.path.exists(CONFIG_FILE_NAME):
        try:
            with open(CONFIG_FILE_NAME, "r", encoding="utf-8") as f: current_full_config = json.load(f)
        except Exception as e:
            print(f"config.json の読み込み中にエラーが発生したため、更新は現在のメモリ上の設定をベースにします: {e}")
            current_full_config = bot_config.copy() # メモリ上の最新設定を使う
    else: current_full_config = bot_config.copy() # メモリ上の最新設定を使う

    current_full_config.update(new_settings)
    try:
        with open(CONFIG_FILE_NAME, "w", encoding="utf-8") as f:
            json.dump(current_full_config, f, indent=4, ensure_ascii=False)
        print(f"設定を '{CONFIG_FILE_NAME}' に保存しました。")
        # グローバル変数も更新
        bot_config.update(new_settings) # bot_config 自体を更新
        UPLOAD_DESTINATION = bot_config.get("upload_destination", DEFAULT_CONFIG["upload_destination"])
        GDRIVE_TARGET_FOLDER_ID = bot_config.get("gdrive_target_folder_id")
        GDRIVE_CREATE_YM_FOLDERS = bot_config.get("gdrive_create_ym_folders", DEFAULT_CONFIG["gdrive_create_ym_folders"])
        new_gdrive_key_path = bot_config.get("gdrive_service_account_key_path", DEFAULT_CONFIG["gdrive_service_account_key_path"])
        
        path_changed = (GDRIVE_SERVICE_ACCOUNT_KEY_PATH != new_gdrive_key_path)
        GDRIVE_SERVICE_ACCOUNT_KEY_PATH = new_gdrive_key_path
        
        # キーパス変更、またはDrive宛でフォルダIDが新規/変更された場合、または宛先がgdriveに変更された場合は再初期化
        should_reinitialize_gdrive = path_changed
        if "gdrive_target_folder_id" in new_settings and new_settings["gdrive_target_folder_id"]:
            should_reinitialize_gdrive = True
        if "upload_destination" in new_settings and new_settings["upload_destination"] == "gdrive":
             # GDrive宛に変更されたが、サービスがまだ初期化されていない場合も初期化
            if not gdrive_service and GDRIVE_TARGET_FOLDER_ID and GDRIVE_SERVICE_ACCOUNT_KEY_PATH:
                 should_reinitialize_gdrive = True


        if should_reinitialize_gdrive:
            print("Google Drive関連の設定が変更されたため、サービスを再初期化します。")
            initialize_gdrive_service()
            
    except Exception as e: print(f"エラー: '{CONFIG_FILE_NAME}' の保存中に問題が発生しました: {e}")

def extract_gdrive_folder_id_from_string(input_string: str) -> str:
    # 標準的なフォルダURL (e.g., https://drive.google.com/drive/folders/FOLDER_ID_HERE)
    match_folders_url = re.search(r"folders/([a-zA-Z0-9_-]{25,})", input_string)
    extracted_id = None # extracted_id を初期化
    if match_folders_url:
        extracted_id = match_folders_url.group(1)
        print(f"URL (folders/) からGoogle DriveフォルダIDを抽出しました: {extracted_id}")
        return extracted_id

    # 共有リンクのURL (e.g., https://drive.google.com/drive/u/0/folders/FOLDER_ID_HERE)
    # 上と同じパターンでカバー可能だが、念のため記載。既に抽出済みならそちらを優先
    match_shared_url = re.search(r"u/\d+/folders/([a-zA-Z0-9_-]{25,})", input_string)
    if match_shared_url:
        extracted_id_shared = match_shared_url.group(1)
        if not extracted_id or extracted_id_shared != extracted_id:
            print(f"URL (shared folders with u/N/) からGoogle DriveフォルダIDを抽出しました: {extracted_id_shared}")
            return extracted_id_shared
        # extracted_id が既にあり、同じなら何もしない

    # URLパラメータからの抽出 (e.g., ?id=FOLDER_ID_HERE, &id=FOLDER_ID_HERE)
    match_id_param = re.search(r"[?&]id=([a-zA-Z0-9_-]{25,})", input_string)
    if match_id_param:
        extracted_id_param = match_id_param.group(1)
        if not extracted_id or extracted_id_param != extracted_id:
            print(f"URLパラメータからGoogle DriveフォルダIDを抽出しました: {extracted_id_param}")
            return extracted_id_param

    if extracted_id: # 上のいずれかで抽出されていればそれを返す
        return extracted_id
        
    # それでも見つからなければ、入力文字列が直接IDであると仮定する
    print(f"入力文字列をそのままGoogle DriveフォルダIDとして扱います: {input_string.strip()}")
    return input_string.strip()

# --- Google Drive API 用ヘルパー関数 ---
async def execute_gdrive_api_call(func, *args, **kwargs):
    """ Google Drive APIの同期的な呼び出しを非同期に実行するラッパー """
    try:
        return await asyncio.to_thread(func, *args, **kwargs)
    except Exception as e:
        print(f"Error executing GDrive API call {func.__name__ if hasattr(func, '__name__') else 'unknown_func'}: {e}")
        return None 

async def get_gdrive_folder_id_by_name(parent_id: str, folder_name: str, service) -> str | None:
    """ 指定された親フォルダIDの下にある特定の名前のフォルダIDを取得 """
    if not service: return None
    def _api_call():
        query = f"mimeType='application/vnd.google-apps.folder' and trashed=false and name='{folder_name}' and '{parent_id}' in parents"
        response = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        folders = response.get('files', [])
        if folders:
            return folders[0].get('id')
        return None
    return await execute_gdrive_api_call(_api_call)

async def list_gdrive_subfolders(parent_id: str, service, name_pattern_re: str | None = None) -> list[dict]:
    """ 指定された親フォルダIDの直下にあるサブフォルダの一覧を返す (ページネーション対応、名前降順ソート) """
    if not service: return []
    folders_found = []
    
    def _api_call_page(page_token_val=None):
        query = f"mimeType='application/vnd.google-apps.folder' and trashed=false and '{parent_id}' in parents"
        return service.files().list(q=query,
                                    spaces='drive',
                                    fields='nextPageToken, files(id, name)',
                                    pageToken=page_token_val).execute()
    page_token = None
    while True:
        response = await execute_gdrive_api_call(_api_call_page, page_token)
        if response is None: break 
        for folder in response.get('files', []):
            if name_pattern_re:
                if re.match(name_pattern_re, folder.get('name')):
                    folders_found.append({'id': folder.get('id'), 'name': folder.get('name')})
            else:
                folders_found.append({'id': folder.get('id'), 'name': folder.get('name')})
        page_token = response.get('nextPageToken', None)
        if page_token is None:
            break
    return sorted(folders_found, key=lambda x: x['name'], reverse=True) # 名前で降順ソート

async def list_files_in_gdrive_folder(folder_id: str, service, keyword: str | None = None) -> list[dict]:
    """ 指定されたGoogle DriveのフォルダID内のファイル一覧を返す (ページネーション対応、名前昇順ソート) """
    if not service: return []
    files_found = []
    def _api_call_page(page_token_val=None):
        query = f"mimeType!='application/vnd.google-apps.folder' and trashed=false and '{folder_id}' in parents"
        if keyword: 
            sanitized_keyword = keyword.replace("'", "\\'") 
            query += f" and name contains '{sanitized_keyword}'"
        return service.files().list(q=query,
                                    spaces='drive',
                                    fields='nextPageToken, files(id, name, createdTime, webViewLink, mimeType, size)', # size も取得
                                    pageToken=page_token_val).execute()
    page_token = None
    while True:
        response = await execute_gdrive_api_call(_api_call_page, page_token)
        if response is None: break 
        for file_item in response.get('files', []):
            files_found.append(file_item)
        page_token = response.get('nextPageToken', None)
        if page_token is None:
            break
    return sorted(files_found, key=lambda x: x.get('name')) # 名前で昇順ソート


async def get_gdrive_file_id_from_filepath(filepath: str, service, base_folder_id: str) -> tuple[str | None, str | None]:
    """
    filepath (YYYYMM/filename.ext) から GDrive の fileId と 年月フォルダID を取得する。
    見つからなければ (None, None) を返す。ファイルが見つかれば (file_id, year_month_folder_id)
    """
    if not service or not base_folder_id:
        return None, None
    try:
        ym_dir_name, filename = filepath.split('/', 1)
        if not (len(ym_dir_name) == 6 and ym_dir_name.isdigit()): # YYYYMM形式チェック
             print(f"get_gdrive_file_id_from_filepath: 年月フォルダ名 '{ym_dir_name}' の形式が不正です。")
             return None, None
    except ValueError:
        print(f"get_gdrive_file_id_from_filepath: filepath '{filepath}' の形式が不正です。")
        return None, None

    year_month_folder_id = await get_gdrive_folder_id_by_name(base_folder_id, ym_dir_name, service)
    if not year_month_folder_id:
        # print(f"get_gdrive_file_id_from_filepath: 年月フォルダ '{ym_dir_name}' が GDrive に見つかりません。")
        return None, year_month_folder_id # 年月フォルダがない場合はファイルもない

    files_in_folder = await list_files_in_gdrive_folder(year_month_folder_id, service, keyword=filename) 
    if files_in_folder:
        for file_item in files_in_folder:
            if file_item.get("name") == filename: 
                return file_item.get("id"), year_month_folder_id
    return None, year_month_folder_id

async def download_gdrive_file_to_bytesio(service, file_id: str) -> io.BytesIO | None:
    """ GDriveからファイルをダウンロードし、BytesIOオブジェクトとして返す """
    if not service or not file_id:
        return None
    try:
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        
        done = False
        while not done:
            status, done = await asyncio.to_thread(downloader.next_chunk)
            if status: # status が None でないことを確認
                 print(f"GDrive Download progress for {file_id}: {int(status.progress() * 100)}%")
        fh.seek(0)
        return fh
    except Exception as e:
        print(f"Error downloading GDrive file {file_id} to memory: {e}")
        return None

# --- 管理者チェック ---
def is_admin():
    async def predicate(interaction: discord.Interaction):
        if interaction.guild is None:
            await interaction.response.send_message("このコマンドはサーバー内でのみ実行可能です。", ephemeral=True)
            return False
        if not ADMIN_ROLE_NAMES:
            await interaction.response.send_message("実行に必要なロールがBOTに設定されていません。BOT管理者にお問い合わせください。", ephemeral=True)
            return False
        author_roles = [role.name for role in interaction.user.roles]
        if any(admin_role in author_roles for admin_role in ADMIN_ROLE_NAMES):
            return True
        else:
            await interaction.response.send_message(f"このコマンドの実行には、次のいずれかのロールが必要です: `{', '.join(ADMIN_ROLE_NAMES)}`", ephemeral=True)
            return False
    return app_commands.check(predicate)

# --- Gemini タグ生成 ---
async def get_tags_from_gemini(file_path, original_filename, mime_type):
    global gemini_model_instance
    if not gemini_model_instance:
        print("Geminiモデルが初期化されていないため、タグ生成をスキップします。")
        return "notags"

    print(f"Gemini APIにファイル '{original_filename}' (MIMEタイプ: {mime_type}) を送信してタグを生成します...")
    uploaded_file_resource = None
    try:
        # display_name は必須ではないが、デバッグ等に役立つ可能性がある
        # mime_type も指定できるはずだが、現状のSDKではupload_fileの引数に直接はない模様。
        # genai.upload_file は内部でファイルタイプを推測するか、汎用的なタイプとして扱う
        uploaded_file_resource = genai.upload_file(path=file_path, display_name=original_filename)
        print(f"Gemini APIにファイル '{original_filename}' (ID: {uploaded_file_resource.name}) をアップロードしました。")

        prompt = load_tagging_prompt()
        response = await gemini_model_instance.generate_content_async(
            [prompt, uploaded_file_resource],
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
        # エラーレスポンスに詳細が含まれているか確認 (例: response.prompt_feedback)
        if hasattr(e, 'response') and hasattr(e.response, 'prompt_feedback'):
            print(f"Gemini API Prompt Feedback: {e.response.prompt_feedback}")
        return "notags"
    finally:
        if uploaded_file_resource and hasattr(uploaded_file_resource, 'name'):
             try:
                 print(f"Gemini APIからアップロードされたファイル '{uploaded_file_resource.name}' の削除を試みます...")
                 genai.delete_file(uploaded_file_resource.name)
                 print(f"Gemini APIからアップロードされたファイル '{uploaded_file_resource.name}' を削除しました。")
             except Exception as e_del:
                 print(f"Gemini APIからアップロードされたファイル {uploaded_file_resource.name} の削除中にエラー: {e_del}")

# --- GDrive フォルダ操作 (同期) ---
def get_or_create_drive_folder(parent_folder_id: str, folder_name: str) -> str | None:
    if not gdrive_service or not google_drive_libs_available:
        print("Driveサービスが利用不可のため、フォルダ操作はできません。")
        return None
    try:
        query = f"mimeType='application/vnd.google-apps.folder' and trashed=false and name='{folder_name}' and '{parent_folder_id}' in parents"
        response = gdrive_service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        folders = response.get('files', [])
        if folders:
            print(f"Driveフォルダ '{folder_name}' が見つかりました (ID: {folders[0].get('id')})。")
            return folders[0].get('id')
        else:
            print(f"Driveフォルダ '{folder_name}' が見つからないため、作成します...")
            file_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [parent_folder_id]
            }
            folder = gdrive_service.files().create(body=file_metadata, fields='id').execute()
            print(f"Driveフォルダ '{folder_name}' を作成しました (ID: {folder.get('id')})。")
            return folder.get('id')
    except Exception as e:
        print(f"Driveフォルダ '{folder_name}' の検索または作成中にエラー: {e}")
        return None

# --- GDrive アップロード (同期呼び出し含む) ---
async def upload_to_gdrive(local_file_path: str, drive_filename: str, attachment_content_type: str) -> dict | None:
    if not gdrive_service or not google_drive_libs_available:
        print("Google Driveサービスが利用できないため、アップロードをスキップします。")
        return None
    if not GDRIVE_TARGET_FOLDER_ID:
        print("Google DriveのターゲットフォルダIDが設定されていません。アップロードをスキップします。")
        return None

    parent_id_to_upload = GDRIVE_TARGET_FOLDER_ID
    if GDRIVE_CREATE_YM_FOLDERS:
        now = datetime.datetime.now()
        year_month_folder_name = now.strftime("%Y%m")
        # get_or_create_drive_folder は同期関数なので、ここではそのまま呼び出す
        # 大量アップロード時にはここも非同期化の検討が必要になるかもしれないが、現状は許容
        ym_drive_folder_id = get_or_create_drive_folder(GDRIVE_TARGET_FOLDER_ID, year_month_folder_name)
        if ym_drive_folder_id:
            parent_id_to_upload = ym_drive_folder_id
        else:
            print(f"年月フォルダ '{year_month_folder_name}' の準備に失敗したため、設定されたメインターゲットフォルダにアップロードします。")

    file_metadata = {'name': drive_filename, 'parents': [parent_id_to_upload]}
    try:
        mime_type = attachment_content_type if attachment_content_type else 'application/octet-stream'
        media = MediaFileUpload(local_file_path, mimetype=mime_type, resumable=True)
        print(f"Google Drive ({parent_id_to_upload}) へ '{drive_filename}' をアップロード開始...")
        
        # gdrive_service.files().create().execute() はブロッキングコール
        # asyncio.to_thread を使って非同期に実行
        uploaded_file = await asyncio.to_thread(
            gdrive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id, name, webViewLink, thumbnailLink, size'
            ).execute
        )
        print(f"ファイル '{uploaded_file.get('name')}' がGoogle Driveにアップロードされました。ID: {uploaded_file.get('id')}, Link: {uploaded_file.get('webViewLink')}")
        return uploaded_file
    except Exception as e:
        print(f"Google Driveへのファイルアップロード中にエラーが発生しました: {e}")
        return None

# --- 確認ビュー (ファイル削除用) ---
class ConfirmDeleteView(discord.ui.View):
    def __init__(self, author_id: int, file_path_to_delete: str, filename_display: str): # file_path_to_delete はローカルパスまたはGDrive ID
        super().__init__(timeout=30.0)
        self.author_id = author_id
        self.file_path_to_delete = file_path_to_delete # 削除対象の識別子
        self.filename_display = filename_display
        self.confirmed: bool | None = None
        self.interaction_message: discord.InteractionMessage | None = None

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.author_id:
            await interaction.response.send_message("この操作はコマンドを実行した本人のみが行えます。", ephemeral=True)
            return False
        return True

    @discord.ui.button(label="削除実行", style=discord.ButtonStyle.danger)
    async def confirm_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.confirmed = True
        for item in self.children:
            item.disabled = True
        # メッセージはコマンド側で編集するので、ここでは汎用的なものに
        await interaction.response.edit_message(content=f"ファイル `{self.filename_display}` の削除処理を準備しています...", view=self)
        self.stop()

    @discord.ui.button(label="キャンセル", style=discord.ButtonStyle.secondary)
    async def cancel_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.confirmed = False
        for item in self.children:
            item.disabled = True
        await interaction.response.edit_message(content=f"ファイル `{self.filename_display}` の削除はキャンセルされました。", view=self)
        self.stop()

    async def on_timeout(self):
        if self.confirmed is None:
            for item in self.children:
                item.disabled = True
            if self.interaction_message:
                try:
                    await self.interaction_message.edit(content=f"ファイル `{self.filename_display}` の削除確認がタイムアウトしました。", view=self)
                except discord.NotFound: pass
                except discord.HTTPException as e: print(f"タイムアウト時のメッセージ編集エラー: {e}")
            self.stop()

# --- BOTイベント ---
@bot.event
async def on_ready():
    global current_gemini_model, UPLOAD_DESTINATION # UPLOAD_DESTINATION もグローバル参照するように
    print(f'{bot.user.name} としてログインしました (ID: {bot.user.id})')
    print(f'監視中のサーバー数: {len(bot.guilds)}')
    print(f'ベースアップロードフォルダ(ローカル): {os.path.abspath(BASE_UPLOAD_FOLDER)}')
    
    # bot_config から最新のアップロード先を読み込む
    UPLOAD_DESTINATION = bot_config.get("upload_destination", DEFAULT_CONFIG["upload_destination"])
    print(f'現在のアップロード先: {UPLOAD_DESTINATION}')
    print(f'Geminiコマンド管理者ロール: {ADMIN_ROLE_NAMES}')
    if gemini_model_instance:
        print(f'使用中Geminiモデル: {current_gemini_model}')
    else:
        print('Geminiモデルは初期化されていません。')
    
    load_tagging_prompt()

    if not os.path.exists(BASE_UPLOAD_FOLDER):
        try: # ベースフォルダ作成失敗時のエラーハンドリング
            os.makedirs(BASE_UPLOAD_FOLDER)
            print(f"ベースフォルダ '{BASE_UPLOAD_FOLDER}' を作成しました。")
        except Exception as e:
            print(f"エラー: ベースフォルダ '{BASE_UPLOAD_FOLDER}' の作成に失敗しました: {e}")


    initialize_gdrive_service() # Google Driveサービスを初期化 (起動時に一度行う)

    try:
        await bot.tree.sync()
        print("スラッシュコマンドを同期しました。")
    except Exception as e:
        print(f"スラッシュコマンドの同期に失敗しました: {e}")
    print('------')

@bot.event
async def on_message(message):
    if message.author == bot.user: return
    if message.attachments:
        ctx = await bot.get_context(message) # サーバー情報などのため
        for attachment in message.attachments:
            allowed_image_types = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp')
            allowed_video_types = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
            file_ext = os.path.splitext(attachment.filename)[1].lower()

            if not (file_ext in allowed_image_types or file_ext in allowed_video_types):
                # await message.channel.send(f"ファイル '{attachment.filename}' の形式 ({file_ext}) はサポートされていません。画像または動画ファイルをアップロードしてください。")
                # サポート外形式はログにのみ残し、ユーザーには通知しない運用も検討 (チャンネルがログで溢れるのを防ぐため)
                print(f"Skipping unsupported file type: {attachment.filename} ({file_ext})")
                continue

            limit_bytes = 8 * 1024 * 1024 
            if ctx.guild and hasattr(ctx.guild, 'filesize_limit'):
                limit_bytes = ctx.guild.filesize_limit
            
            if attachment.size > limit_bytes:
                 await message.channel.send(f"ファイル '{attachment.filename}' ({attachment.size // 1024 // 1024}MB) はサイズが大きすぎます (サーバー上限: {limit_bytes // 1024 // 1024}MB)。")
                 continue

            temp_dir = os.path.join(BASE_UPLOAD_FOLDER, "temp")
            if not os.path.exists(temp_dir):
                try: os.makedirs(temp_dir)
                except Exception as e:
                    print(f"一時フォルダ '{temp_dir}' の作成に失敗: {e}")
                    await message.channel.send(f"'{attachment.filename}' の処理中に内部エラーが発生しました（一時フォルダ作成不可）。")
                    continue
            
            temp_save_path = os.path.join(temp_dir, f"temp_{attachment.id}_{sanitize_filename_component(attachment.filename)}")
            
            try:
                await attachment.save(temp_save_path)
            except Exception as e_save:
                print(f"一時ファイル '{temp_save_path}' の保存に失敗: {e_save}")
                await message.channel.send(f"ファイル '{attachment.filename}' の一時保存に失敗しました。")
                continue

            processing_msg = await message.channel.send(f"ファイル '{attachment.filename}' を処理中... 自動タグ付けを開始します。")
            
            tags_str = "notags"
            if gemini_model_instance:
                try:
                    if file_ext in allowed_image_types:
                        try:
                            img = Image.open(temp_save_path)
                            img.verify()
                            img.close()
                        except Exception as img_err:
                            await processing_msg.edit(content=f"ファイル '{attachment.filename}' は有効な画像ではないようです。処理を中断します。({img_err})")
                            if os.path.exists(temp_save_path): os.remove(temp_save_path)
                            continue
                    
                    tags_str = await get_tags_from_gemini(temp_save_path, attachment.filename, attachment.content_type)
                except Exception as e:
                    print(f"タグ付け処理中にエラー: {e}")
                    await processing_msg.edit(content=f"ファイル '{attachment.filename}' のタグ付け中にエラーが発生しました。タグなしで処理を続行します。")
                    tags_str = "notags"
            else:
                await processing_msg.edit(content=f"ファイル '{attachment.filename}' を処理中... (Gemini API未設定のためタグ付けスキップ)")

            date_str = datetime.datetime.now().strftime("%Y%m%d")
            original_filename_no_ext, original_ext = os.path.splitext(attachment.filename)
            sanitized_original_filename = sanitize_filename_component(original_filename_no_ext)
            new_filename = f"{date_str}_{tags_str}_{sanitized_original_filename}{original_ext}"
            
            display_tags_on_message = tags_str.replace("_", "-") if tags_str != "notags" else "なし"
            
            # 現在のアップロード先をbot_configから再取得（コマンドで変更された場合に対応）
            current_upload_dest_on_message = bot_config.get("upload_destination", DEFAULT_CONFIG["upload_destination"])

            if current_upload_dest_on_message == "gdrive":
                if gdrive_service and GDRIVE_TARGET_FOLDER_ID:
                    gdrive_file_info = await upload_to_gdrive(temp_save_path, new_filename, attachment.content_type)
                    if gdrive_file_info:
                        file_link = gdrive_file_info.get('webViewLink', 'リンク不明')
                        await processing_msg.edit(content=(
                            f"ファイル '{attachment.filename}' をGoogle Driveにアップロードし、'{new_filename}' として保存しました。\n"
                            f"自動タグ: `{display_tags_on_message}`\nリンク: <{file_link}>"
                        ))
                    else:
                        await processing_msg.edit(content=f"ファイル '{attachment.filename}' のGoogle Driveへのアップロードに失敗しました。ローカルにも保存されませんでした。")
                else:
                    await processing_msg.edit(content=f"Google Driveが設定されていないか、サービスが利用できないため、'{attachment.filename}' のアップロードをスキップしました。ローカルにも保存されません。")
                
                if os.path.exists(temp_save_path):
                    try: os.remove(temp_save_path); print(f"一時ファイル '{temp_save_path}' を削除しました。")
                    except Exception as e_rm: print(f"一時ファイル '{temp_save_path}' の削除失敗: {e_rm}")

            elif current_upload_dest_on_message == "local":
                local_ym_folder = create_year_month_folder_if_not_exists(BASE_UPLOAD_FOLDER)
                final_save_path = os.path.join(local_ym_folder, new_filename)
                try:
                    os.rename(temp_save_path, final_save_path)
                    print(f"ファイル '{attachment.filename}' を '{final_save_path}' に保存しました。")
                    await processing_msg.edit(content=(
                        f"ファイル '{attachment.filename}' をローカルに保存しました: '{new_filename}'\n自動タグ: `{display_tags_on_message}`"
                    ))
                except Exception as e:
                    print(f"ローカル保存エラー: {e}")
                    await processing_msg.edit(content=f"'{attachment.filename}' のローカル保存中にエラーが発生しました。")
                    if os.path.exists(temp_save_path): # rename失敗時は一時ファイルを削除
                        try: os.remove(temp_save_path); print(f"エラー発生のため一時ファイル '{temp_save_path}' を削除しました。")
                        except Exception as e_rm: print(f"一時ファイル '{temp_save_path}' の削除失敗: {e_rm}")
            else:
                print(f"不明なアップロード先が設定されています: {current_upload_dest_on_message}")
                await processing_msg.edit(content=f"アップロード先の設定が不明なため、'{attachment.filename}' の処理を中断しました。")
                if os.path.exists(temp_save_path):
                    try: os.remove(temp_save_path); print(f"不明なアップロード先のため一時ファイル '{temp_save_path}' を削除しました。")
                    except Exception as e_rm: print(f"一時ファイル '{temp_save_path}' の削除失敗: {e_rm}")
            
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
    except Exception as e:
        print(f"Geminiモデルのオートコンプリート中にエラー: {e}")
        return []
    return choices

async def year_month_autocomplete(interaction: discord.Interaction, current: str) -> list[app_commands.Choice[str]]:
    choices = []
    current_upload_dest = bot_config.get("upload_destination", DEFAULT_CONFIG["upload_destination"])

    if current_upload_dest == "local":
        ym_folders = set()
        if not os.path.exists(BASE_UPLOAD_FOLDER): return []
        try:
            for item in os.listdir(BASE_UPLOAD_FOLDER):
                if os.path.isdir(os.path.join(BASE_UPLOAD_FOLDER, item)) and len(item) == 6 and item.isdigit():
                    ym_folders.add(item)
            
            for folder_name in sorted(list(ym_folders), reverse=True): 
                if current.lower() in folder_name.lower():
                    choices.append(app_commands.Choice(name=folder_name, value=folder_name))
                if len(choices) >= 25: break
        except Exception as e:
            print(f"year_month_autocomplete (local) 中にエラー: {e}")
            return [] # エラー時は空を返す
            
    elif current_upload_dest == "gdrive":
        if not gdrive_service or not GDRIVE_TARGET_FOLDER_ID:
            print("year_month_autocomplete (gdrive): GDriveサービス未初期化またはターゲットフォルダID未設定")
            return []
        try:
            subfolders = await list_gdrive_subfolders(GDRIVE_TARGET_FOLDER_ID, gdrive_service, name_pattern_re=r"^\d{6}$")
            if subfolders: # list_gdrive_subfolders は名前降順でソート済み
                for folder_info in subfolders:
                    folder_name = folder_info['name']
                    if current.lower() in folder_name.lower():
                        choices.append(app_commands.Choice(name=folder_name, value=folder_name))
                    if len(choices) >= 25: break
        except Exception as e:
            print(f"year_month_autocomplete (gdrive) 中にエラー: {e}")
            return [] # エラー時は空を返す
            
    return choices

# [bot.py の修正箇所]

async def filename_autocomplete(interaction: discord.Interaction, current: str) -> list[app_commands.Choice[str]]:
    choices = []
    specific_ym_folder_name = None 
    current_filename_part_to_search = current
    # bot_config はグローバル変数を直接参照するか、load_bot_config() で毎回読み込むか検討
    # ここでは前回提示したコードの通り、グローバル bot_config を参照する想定
    current_upload_dest = bot_config.get("upload_destination", DEFAULT_CONFIG["upload_destination"])

    if '/' in current and len(current.split('/')[0]) == 6 and current.split('/')[0].isdigit():
        parts = current.split('/', 1)
        specific_ym_folder_name = parts[0]
        current_filename_part_to_search = parts[1] if len(parts) > 1 else ""

    if current_upload_dest == "local":
        folders_to_search_paths = []
        if specific_ym_folder_name:
            path_to_check = os.path.join(BASE_UPLOAD_FOLDER, specific_ym_folder_name)
            if os.path.isdir(path_to_check):
                folders_to_search_paths.append(path_to_check)
        else: 
            if os.path.exists(BASE_UPLOAD_FOLDER):
                for item in sorted(os.listdir(BASE_UPLOAD_FOLDER), reverse=True): 
                    item_path = os.path.join(BASE_UPLOAD_FOLDER, item)
                    if os.path.isdir(item_path) and len(item) == 6 and item.isdigit():
                        folders_to_search_paths.append(item_path)
        
        for folder_path in folders_to_search_paths:
            year_month_dir_name = os.path.basename(folder_path) # "YYYYMM"
            try:
                for fname in sorted(os.listdir(folder_path)): 
                    if os.path.isfile(os.path.join(folder_path, fname)):
                        if current_filename_part_to_search.lower() in fname.lower():
                            # --- 表示名 (name) の生成 ---
                            suffix = f" (in {year_month_dir_name})"
                            allowed_fname_len_for_display = 100 - len(suffix) 
                            display_fname = fname
                            if len(fname) > allowed_fname_len_for_display:
                                display_fname = fname[:max(0, allowed_fname_len_for_display - 3)] + "..."
                            
                            final_choice_name = f"{display_fname}{suffix}"
                            if len(final_choice_name) > 100: # 更なる最終チェック
                                final_choice_name = final_choice_name[:97] + "..."
                            
                            # --- 値 (value) の生成と調整 ---
                            base_value = f"{year_month_dir_name}/{fname}"
                            value_to_set = base_value
                            if len(base_value) > 100:
                                prefix = f"{year_month_dir_name}/"
                                # prefix の7文字 + ファイル名の最大長
                                max_fname_len_for_value = 100 - len(prefix)
                                if max_fname_len_for_value < 1: # ほぼありえないが YYYYMM/ が長すぎる場合
                                    value_to_set = base_value[:100] # 単純に先頭100文字
                                else:
                                    value_to_set = f"{prefix}{fname[:max_fname_len_for_value]}"
                            
                            choices.append(app_commands.Choice(name=final_choice_name, value=value_to_set))
                            if len(choices) >= 25: break
            except Exception as e:
                print(f"filename_autocomplete (local) でフォルダ '{folder_path}' のスキャン中にエラー: {e}")
            if len(choices) >= 25: break
            
    elif current_upload_dest == "gdrive":
        if not gdrive_service or not GDRIVE_TARGET_FOLDER_ID:
            print("filename_autocomplete (gdrive): GDriveサービス未初期化またはターゲットフォルダID未設定")
            return []

        gdrive_folders_to_scan_info = [] 
        try:
            if specific_ym_folder_name:
                ym_folder_id = await get_gdrive_folder_id_by_name(GDRIVE_TARGET_FOLDER_ID, specific_ym_folder_name, gdrive_service)
                if ym_folder_id:
                    gdrive_folders_to_scan_info.append({'id': ym_folder_id, 'name': specific_ym_folder_name})
            else:
                subfolders = await list_gdrive_subfolders(GDRIVE_TARGET_FOLDER_ID, gdrive_service, name_pattern_re=r"^\d{6}$")
                gdrive_folders_to_scan_info.extend(subfolders)

            for folder_info in gdrive_folders_to_scan_info:
                folder_id_to_scan = folder_info['id']
                current_year_month_name = folder_info['name'] # "YYYYMM"
                
                files_in_gdrive = await list_files_in_gdrive_folder(folder_id_to_scan, gdrive_service, keyword=current_filename_part_to_search)
                if files_in_gdrive: 
                    for gfile in files_in_gdrive:
                        gfile_name = gfile.get("name")
                        if not gfile_name: continue
                        
                        # --- 表示名 (name) の生成 ---
                        suffix = f" (in {current_year_month_name})"
                        allowed_gfname_len_for_display = 100 - len(suffix)
                        display_gfname = gfile_name
                        if len(gfile_name) > allowed_gfname_len_for_display:
                             display_gfname = gfile_name[:max(0, allowed_gfname_len_for_display - 3)] + "..."

                        final_choice_name = f"{display_gfname}{suffix}"
                        if len(final_choice_name) > 100: # 更なる最終チェック
                            final_choice_name = final_choice_name[:97] + "..."

                        # --- 値 (value) の生成と調整 ---
                        base_value = f"{current_year_month_name}/{gfile_name}"
                        value_to_set = base_value
                        if len(base_value) > 100:
                            prefix = f"{current_year_month_name}/"
                            max_gfname_len_for_value = 100 - len(prefix)
                            if max_gfname_len_for_value < 1:
                                value_to_set = base_value[:100]
                            else:
                                value_to_set = f"{prefix}{gfile_name[:max_gfname_len_for_value]}"
                        
                        choices.append(app_commands.Choice(name=final_choice_name, value=value_to_set))
                        if len(choices) >= 25: break
                if len(choices) >= 25: break
        except Exception as e:
            print(f"filename_autocomplete (gdrive) 中にエラー: {e}")
            return []
            
    return choices

# --- コマンドグループの定義 ---
gemini_group = app_commands.Group(name="gemini", description="Geminiモデル関連の操作を行います。")
files_group = app_commands.Group(name="files", description="アップロードされたファイルの管理を行います。")
upload_settings_group = app_commands.Group(name="upload_settings", description="ファイルのアップロード先設定を管理します。")

# --- スラッシュコマンド ---
@bot.tree.command(name="upload_guide", description="ファイルアップロード方法の案内")
async def upload_guide(interaction: discord.Interaction):
    await interaction.response.send_message(
        "ファイルをアップロードするには、このチャンネルに直接ファイルをドラッグ＆ドロップするか、メッセージ入力欄の「+」ボタンからファイルを添付して送信してください。\n"
        "画像または動画ファイルが対象です。自動的にタグが付けられ、設定に応じて保存されます。"
    )

# --- /files サブコマンド ---
@files_group.command(name="list", description="保存されているファイルの一覧を表示します。")
@app_commands.describe(year_month="表示する年月 (例: 202305)。", keyword="ファイル名やタグに含まれるキーワードで絞り込みます。")
@app_commands.autocomplete(year_month=year_month_autocomplete)
async def files_list(interaction: discord.Interaction, year_month: str = None, keyword: str = None):
    await interaction.response.defer()
    found_files_details = [] 
    current_upload_dest = bot_config.get("upload_destination", DEFAULT_CONFIG["upload_destination"])

    if current_upload_dest == "local":
        search_paths = [] 
        if year_month:
            if not (len(year_month) == 6 and year_month.isdigit()):
                await interaction.followup.send("年月の指定が正しくありません。YYYYMM形式で入力してください (例: 202305)。")
                return
            target_ym_folder = os.path.join(BASE_UPLOAD_FOLDER, year_month)
            if os.path.exists(target_ym_folder) and os.path.isdir(target_ym_folder):
                search_paths.append(target_ym_folder)
            else:
                await interaction.followup.send(f"指定された年月フォルダ '{year_month}' はローカルに見つかりません。")
                return
        else:
            if os.path.exists(BASE_UPLOAD_FOLDER):
                for item in sorted(os.listdir(BASE_UPLOAD_FOLDER), reverse=True): 
                    item_path = os.path.join(BASE_UPLOAD_FOLDER, item)
                    if os.path.isdir(item_path) and len(item) == 6 and item.isdigit():
                        search_paths.append(item_path)
            
        if not search_paths:
            msg = "検索対象のローカルフォルダが見つかりません。"
            if year_month: msg = f"指定された年月フォルダ '{year_month}' はローカルに見つかりません。"
            elif not os.path.exists(BASE_UPLOAD_FOLDER): msg = f"ベースアップロードフォルダ '{BASE_UPLOAD_FOLDER}' がローカルに見つかりません。"
            await interaction.followup.send(msg)
            return

        for folder_to_scan in search_paths:
            try:
                current_year_month_name = os.path.basename(folder_to_scan)
                for fname in sorted(os.listdir(folder_to_scan)):
                    if os.path.isfile(os.path.join(folder_to_scan, fname)):
                        if keyword and keyword.lower() not in fname.lower():
                            continue
                        parsed_info = parse_bot_filename(fname)
                        found_files_details.append({
                            "fullname": fname, "date": parsed_info["date"], 
                            "tags": parsed_info["tags_display"], 
                            "original_name": parsed_info["original_stem"], 
                            "year_month": current_year_month_name
                        })
            except Exception as e:
                print(f"ローカルフォルダ '{folder_to_scan}' のスキャン中にエラー: {e}")
                
    elif current_upload_dest == "gdrive":
        if not gdrive_service:
            await interaction.followup.send("Google Driveサービスが初期化されていません。設定を確認してください。")
            return
        if not GDRIVE_TARGET_FOLDER_ID:
            await interaction.followup.send("Google DriveのメインターゲットフォルダIDが設定されていません。")
            return

        gdrive_folders_to_scan_info = [] 
        if year_month:
            if not (len(year_month) == 6 and year_month.isdigit()):
                await interaction.followup.send("年月の指定が正しくありません。YYYYMM形式で入力してください (例: 202305)。")
                return
            
            ym_folder_id = await get_gdrive_folder_id_by_name(GDRIVE_TARGET_FOLDER_ID, year_month, gdrive_service)
            if ym_folder_id:
                gdrive_folders_to_scan_info.append({'id': ym_folder_id, 'name': year_month})
            else:
                await interaction.followup.send(f"指定された年月フォルダ '{year_month}' はGoogle Drive上に見つかりません。")
                return
        else:
            subfolders = await list_gdrive_subfolders(GDRIVE_TARGET_FOLDER_ID, gdrive_service, name_pattern_re=r"^\d{6}$")
            gdrive_folders_to_scan_info.extend(subfolders) 
        
        if not gdrive_folders_to_scan_info:
            msg = "検索対象の年月フォルダがGoogle Drive上に見つかりません。"
            if year_month: msg = f"指定された年月フォルダ '{year_month}' はGoogle Drive上に見つかりません。"
            await interaction.followup.send(msg)
            return

        for folder_info in gdrive_folders_to_scan_info:
            try:
                files_in_gdrive = await list_files_in_gdrive_folder(folder_info['id'], gdrive_service, keyword=keyword)
                if files_in_gdrive is None: 
                    print(f"Google Driveフォルダ '{folder_info['name']}' (ID: {folder_info['id']}) のファイル一覧取得に失敗しました。")
                    continue
                for gfile in files_in_gdrive:
                    gfile_name = gfile.get("name")
                    if not gfile_name: continue
                    parsed_info = parse_bot_filename(gfile_name)
                    found_files_details.append({
                        "fullname": gfile_name, "date": parsed_info["date"], 
                        "tags": parsed_info["tags_display"], 
                        "original_name": parsed_info["original_stem"], 
                        "year_month": folder_info['name'], 
                        "gdrive_id": gfile.get("id"), 
                        "gdrive_link": gfile.get("webViewLink") 
                    })
            except Exception as e:
                print(f"Google Driveフォルダ '{folder_info['name']}' の処理中にエラー: {e}")

    # --- 共通のEmbed作成・送信処理 ---
    if not found_files_details:
        message = "ファイルは見つかりませんでした。"
        if year_month: message += f" (年月: {year_month})"
        if keyword: message += f" (キーワード: {keyword})"
        await interaction.followup.send(message)
        return

    embed = discord.Embed(title="ファイル一覧", color=discord.Color.blue())
    description_parts = []
    if year_month: description_parts.append(f"年月: `{year_month}`")
    if keyword: description_parts.append(f"キーワード: `{keyword}`")
    if description_parts:
        embed.description = "絞り込み条件: " + " | ".join(description_parts)
    
    embed.set_footer(text=f"アップロード先: {current_upload_dest}")

    MAX_FILES_IN_EMBED = 10 
    for i, file_info in enumerate(found_files_details):
        if i >= MAX_FILES_IN_EMBED:
            embed.add_field(name="...", value=f"他 {len(found_files_details) - MAX_FILES_IN_EMBED} 件のファイルがあります。", inline=False)
            break
        
        field_name = f"📁 `{file_info['fullname']}`"
        field_value = (f"元ファイル名: `{file_info['original_name']}`\n"
                       f"タグ: `{file_info['tags']}`\n"
                       f"保存日: `{file_info['date']}` (in `{file_info['year_month']}`)")
        if current_upload_dest == "gdrive" and file_info.get('gdrive_link'):
             field_value += f"\n[Google Driveで開く]({file_info['gdrive_link']})"
        embed.add_field(name=field_name, value=field_value, inline=False)

    if not embed.fields: 
        await interaction.followup.send("表示できるファイル情報がありません。")
        return
    await interaction.followup.send(embed=embed)


@files_group.command(name="info", description="指定された保存済みファイルの詳細情報を表示します。")
@app_commands.describe(filepath="情報を表示するファイル (年月フォルダ/ファイル名)")
@app_commands.autocomplete(filepath=filename_autocomplete)
async def files_info(interaction: discord.Interaction, filepath: str):
    await interaction.response.defer()
    current_upload_dest = bot_config.get("upload_destination", DEFAULT_CONFIG["upload_destination"])

    try:
        ym_dir_name, filename = filepath.split('/', 1)
    except ValueError:
        await interaction.followup.send("ファイルパスの形式が正しくありません。YYYYMM/ファイル名の形式で入力してください。", ephemeral=True)
        return

    if current_upload_dest == "local":
        full_path = os.path.join(BASE_UPLOAD_FOLDER, ym_dir_name, filename)
        if not os.path.exists(full_path) or not os.path.isfile(full_path):
            await interaction.followup.send(f"ファイル `{filepath}` がローカルに見つかりません。")
            return
        try:
            parsed_info = parse_bot_filename(filename)
            file_size_bytes = os.path.getsize(full_path)
            file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
            
            embed = discord.Embed(title=f"ファイル情報 (ローカル): {filename}", color=discord.Color.green())
            # embed.add_field(name="フルパス (サーバー上)", value=f"`{full_path}`", inline=False) # 表示しない方が安全な場合もある
            embed.add_field(name="ファイル名", value=f"`{filename}`", inline=False)
            embed.add_field(name="年月フォルダ", value=f"`{ym_dir_name}`", inline=True)
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
        except Exception as e:
            print(f"/files info (local) 処理中にエラー: {e}")
            await interaction.followup.send(f"ローカルファイル情報の取得中にエラーが発生しました: {e}")

    elif current_upload_dest == "gdrive":
        if not gdrive_service or not GDRIVE_TARGET_FOLDER_ID:
            await interaction.followup.send("Google Driveサービスが利用できないか、ターゲットフォルダが設定されていません。", ephemeral=True)
            return

        gdrive_file_id, _ = await get_gdrive_file_id_from_filepath(filepath, gdrive_service, GDRIVE_TARGET_FOLDER_ID)

        if not gdrive_file_id:
            await interaction.followup.send(f"ファイル `{filepath}` がGoogle Driveに見つかりません。")
            return
        
        try:
            file_metadata = await execute_gdrive_api_call(
                gdrive_service.files().get(fileId=gdrive_file_id, fields="id, name, mimeType, size, createdTime, modifiedTime, webViewLink, description, parents").execute
            )
            if not file_metadata:
                await interaction.followup.send(f"Google Driveからファイル `{filename}` のメタデータの取得に失敗しました。")
                return

            gdrive_actual_filename = file_metadata.get("name", filename)
            parsed_info = parse_bot_filename(gdrive_actual_filename)
            file_size_bytes = int(file_metadata.get("size", 0))
            file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
            
            embed = discord.Embed(title=f"ファイル情報 (Google Drive): {gdrive_actual_filename}", color=discord.Color.dark_green())
            embed.add_field(name="ファイル名 (Drive上)", value=f"`{gdrive_actual_filename}`", inline=False)
            if file_metadata.get('webViewLink'):
                embed.add_field(name="Google Driveリンク", value=f"[ファイルを開く]({file_metadata.get('webViewLink')})", inline=False)
            embed.add_field(name="年月フォルダ", value=f"`{ym_dir_name}`", inline=True)
            embed.add_field(name="ファイルサイズ", value=f"{file_size_bytes} Bytes ({file_size_mb} MB)", inline=True)
            embed.add_field(name="MIMEタイプ", value=f"`{file_metadata.get('mimeType', '不明')}`", inline=True)
            embed.add_field(name="元ファイル名 (拡張子除く)", value=f"`{parsed_info['original_stem']}`", inline=False)
            embed.add_field(name="拡張子", value=f"`{parsed_info['extension']}`", inline=True)
            embed.add_field(name="抽出されたタグ", value=f"`{parsed_info['tags_display']}`", inline=True)
            embed.add_field(name="抽出された日付", value=f"`{parsed_info['date']}`", inline=True)
            
            if file_metadata.get('createdTime'):
                cdt = datetime.datetime.fromisoformat(file_metadata.get('createdTime').replace('Z', '+00:00'))
                embed.add_field(name="作成日時 (Drive)", value=cdt.strftime('%Y-%m-%d %H:%M:%S %Z'), inline=True)
            if file_metadata.get('modifiedTime'):
                mdt = datetime.datetime.fromisoformat(file_metadata.get('modifiedTime').replace('Z', '+00:00'))
                embed.add_field(name="最終更新日時 (Drive)", value=mdt.strftime('%Y-%m-%d %H:%M:%S %Z'), inline=True)
            if file_metadata.get('description'):
                 embed.add_field(name="説明 (Drive)", value=file_metadata.get('description'), inline=False)

            await interaction.followup.send(embed=embed)
        except Exception as e:
            print(f"/files info (gdrive) 処理中にエラー: {e}")
            await interaction.followup.send(f"Google Driveファイル情報の取得中にエラーが発生しました: {e}")

@files_group.command(name="delete", description="指定された保存済みファイルをサーバーから削除します。")
@app_commands.describe(filepath="削除するファイル (年月フォルダ/ファイル名)")
@app_commands.autocomplete(filepath=filename_autocomplete)
async def files_delete(interaction: discord.Interaction, filepath: str):
    await interaction.response.defer() 
    current_upload_dest = bot_config.get("upload_destination", DEFAULT_CONFIG["upload_destination"])

    try:
        ym_dir_name, filename_to_delete_display = filepath.split('/', 1)
    except ValueError:
        await interaction.followup.send("ファイルパスの形式が正しくありません。YYYYMM/ファイル名の形式で入力してください。", ephemeral=True)
        return

    identifier_for_delete = None # ローカルパスまたはGDrive ID
    delete_target_description = "" # メッセージ用

    if current_upload_dest == "local":
        full_path = os.path.join(BASE_UPLOAD_FOLDER, ym_dir_name, filename_to_delete_display)
        if not os.path.exists(full_path) or not os.path.isfile(full_path):
            await interaction.followup.send(f"ファイル `{filepath}` がローカルに見つかりません。")
            return
        identifier_for_delete = full_path
        delete_target_description = "ローカル"
        
    elif current_upload_dest == "gdrive":
        if not gdrive_service or not GDRIVE_TARGET_FOLDER_ID:
            await interaction.followup.send("Google Driveサービスが利用できないか、ターゲットフォルダが設定されていません。", ephemeral=True)
            return
        gdrive_file_id, _ = await get_gdrive_file_id_from_filepath(filepath, gdrive_service, GDRIVE_TARGET_FOLDER_ID)
        if not gdrive_file_id:
            await interaction.followup.send(f"ファイル `{filepath}` がGoogle Driveに見つかりません。")
            return
        identifier_for_delete = gdrive_file_id
        delete_target_description = "Google Drive"
    
    else: #ありえないが念のため
        await interaction.followup.send("不明なアップロード先です。処理を中断しました。", ephemeral=True)
        return

    if not identifier_for_delete: #念のため
        await interaction.followup.send("削除対象の特定に失敗しました。", ephemeral=True)
        return

    view = ConfirmDeleteView(author_id=interaction.user.id, file_path_to_delete=identifier_for_delete, filename_display=filename_to_delete_display)
    interaction_message = await interaction.followup.send(
        f"**警告 ({delete_target_description}):** ファイル `{filename_to_delete_display}` を本当に削除しますか？この操作は取り消せません。(実行者: {interaction.user.mention})", 
        view=view
    )
    view.interaction_message = interaction_message
    await view.wait()

    if view.confirmed is True:
        try:
            if current_upload_dest == "local":
                os.remove(identifier_for_delete) # identifier_for_delete は full_path
                print(f"ユーザー {interaction.user} によってローカルファイル {identifier_for_delete} が削除されました。")
            elif current_upload_dest == "gdrive":
                await execute_gdrive_api_call(
                    gdrive_service.files().delete(fileId=identifier_for_delete).execute # identifier_for_delete は gdrive_file_id
                )
                print(f"ユーザー {interaction.user} によってGDriveファイル {identifier_for_delete} (元名: {filename_to_delete_display}) が削除されました。")
            
            await interaction_message.edit(content=f"ファイル `{filename_to_delete_display}` ({delete_target_description}) を削除しました。(実行者: {interaction.user.mention})", view=None)
        except Exception as e:
            print(f"{delete_target_description} ファイル削除エラー ({identifier_for_delete}): {e}")
            await interaction_message.edit(content=f"ファイル `{filename_to_delete_display}` ({delete_target_description}) の削除中にエラーが発生しました: {e}", view=None)
    # キャンセルまたはタイムアウトの場合は、view側でメッセージが編集済


@files_group.command(name="get", description="指定された保存済みファイルを取得します。")
@app_commands.describe(filepath="取得するファイル (年月フォルダ/ファイル名)")
@app_commands.autocomplete(filepath=filename_autocomplete)
async def files_get(interaction: discord.Interaction, filepath: str):
    await interaction.response.defer()
    current_upload_dest = bot_config.get("upload_destination", DEFAULT_CONFIG["upload_destination"])

    try:
        ym_dir_name, filename_to_get = filepath.split('/', 1)
    except ValueError:
        await interaction.followup.send("ファイルパスの形式が正しくありません。YYYYMM/ファイル名の形式で入力してください。", ephemeral=True)
        return

    if current_upload_dest == "local":
        full_path = os.path.join(BASE_UPLOAD_FOLDER, ym_dir_name, filename_to_get)
        if not os.path.exists(full_path) or not os.path.isfile(full_path):
            await interaction.followup.send(f"ファイル `{filepath}` がローカルに見つかりません。")
            return
        
        limit_bytes = interaction.guild.filesize_limit if interaction.guild else (8 * 1024 * 1024)
        file_size_bytes = os.path.getsize(full_path)
        if file_size_bytes > limit_bytes:
            await interaction.followup.send(
                f"ファイル `{filename_to_get}` ({round(file_size_bytes / (1024*1024), 2)} MB) はDiscordの送信サイズ上限 ({round(limit_bytes / (1024*1024), 2)} MB) を超えています。"
            )
            return
        try:
            discord_file = discord.File(full_path, filename=filename_to_get)
            await interaction.followup.send(f"ファイル `{filename_to_get}` (ローカル) を送信します: (要求者: {interaction.user.mention})", file=discord_file)
        except Exception as e:
            print(f"ローカルファイル送信エラー ({full_path}): {e}")
            await interaction.followup.send(f"ファイル `{filename_to_get}` の送信中にエラーが発生しました: {e}")

    elif current_upload_dest == "gdrive":
        if not gdrive_service or not GDRIVE_TARGET_FOLDER_ID:
            await interaction.followup.send("Google Driveサービスが利用できないか、ターゲットフォルダが設定されていません。", ephemeral=True)
            return

        gdrive_file_id, _ = await get_gdrive_file_id_from_filepath(filepath, gdrive_service, GDRIVE_TARGET_FOLDER_ID)
        if not gdrive_file_id:
            await interaction.followup.send(f"ファイル `{filepath}` がGoogle Driveに見つかりません。")
            return

        try:
            gfile_meta = await execute_gdrive_api_call(
                gdrive_service.files().get(fileId=gdrive_file_id, fields="size, name").execute
            )
            if not gfile_meta:
                 await interaction.followup.send(f"ファイル `{filename_to_get}` のメタデータ取得に失敗しました。")
                 return
            
            gdrive_actual_filename = gfile_meta.get("name", filename_to_get)
            file_size_bytes = int(gfile_meta.get("size", "0"))
            limit_bytes = interaction.guild.filesize_limit if interaction.guild else (8 * 1024 * 1024)

            if file_size_bytes > limit_bytes:
                await interaction.followup.send(
                    f"ファイル `{gdrive_actual_filename}` ({round(file_size_bytes / (1024*1024), 2)} MB) はDiscordの送信サイズ上限 ({round(limit_bytes / (1024*1024), 2)} MB) を超えています。"
                )
                return
            
            file_bytes_io = await download_gdrive_file_to_bytesio(gdrive_service, gdrive_file_id)
            if not file_bytes_io:
                await interaction.followup.send(f"ファイル `{gdrive_actual_filename}` のGoogle Driveからのダウンロードに失敗しました。")
                return
            
            discord_gdrive_file = discord.File(file_bytes_io, filename=gdrive_actual_filename)
            await interaction.followup.send(f"ファイル `{gdrive_actual_filename}` (Google Drive) を送信します: (要求者: {interaction.user.mention})", file=discord_gdrive_file)

        except Exception as e:
            print(f"Google Driveファイル送信エラー (ID: {gdrive_file_id}): {e}")
            await interaction.followup.send(f"ファイル `{filepath}` の送信中にエラーが発生しました: {e}")

# --- /gemini サブコマンド ---
@gemini_group.command(name="list", description="利用可能なGeminiモデルの一覧を表示します。(ロール制限あり)")
@is_admin()
async def gemini_list(interaction: discord.Interaction):
    if not GEMINI_API_KEY:
        await interaction.response.send_message("Gemini APIキーが設定されていません。", ephemeral=True)
        return
    if not genai: 
        await interaction.response.send_message("Geminiライブラリが利用できません。", ephemeral=True)
        return
        
    await interaction.response.defer(ephemeral=True)
    try:
        models_info_parts = ["利用可能なGeminiモデル (generateContentサポート):\n"]
        count = 0
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                model_display_name = model.name.replace("models/", "")
                current_part = f"- `{model_display_name}` ({model.display_name})\n"
                if len("".join(models_info_parts)) + len(current_part) > 1900: 
                    await interaction.followup.send("".join(models_info_parts), ephemeral=True)
                    models_info_parts = [current_part] 
                else:
                    models_info_parts.append(current_part)
                count += 1
        
        if count == 0 and len(models_info_parts) == 1 and models_info_parts[0].endswith(":\n"):
             models_info_parts.append("generateContentをサポートするGeminiモデルが見つかりませんでした。")

        if models_info_parts: 
            final_message = "".join(models_info_parts)
            # 空でない、かつ初期メッセージのみでない場合に送信
            if final_message.strip() and not (count == 0 and final_message.strip().endswith(":\n") and len(final_message.splitlines()) ==1) :
                await interaction.followup.send(final_message, ephemeral=True)
            elif count == 0 : # generateContentサポートモデルが一つもなかった場合
                await interaction.followup.send("generateContentをサポートする利用可能なGeminiモデルが見つかりませんでした。",ephemeral=True)

    except Exception as e:
        await interaction.followup.send(f"モデル一覧の取得中にエラーが発生しました: {e}", ephemeral=True)

@gemini_group.command(name="set", description="自動タグ付けに使用するGeminiモデルを設定します。(ロール制限あり)")
@app_commands.describe(model_name="Geminiモデル名 (例: gemini-1.5-flash-latest)。")
@app_commands.autocomplete(model_name=gemini_model_autocomplete)
@is_admin()
async def gemini_set(interaction: discord.Interaction, model_name: str):
    global current_gemini_model, gemini_model_instance
    if not GEMINI_API_KEY:
        await interaction.response.send_message("Gemini APIキーが設定されていません。", ephemeral=True)
        return
    if not genai:
        await interaction.response.send_message("Geminiライブラリが利用できません。", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True)
    try:
        full_model_name_to_check = model_name if model_name.startswith("models/") else f"models/{model_name}"
        retrieved_model = genai.get_model(full_model_name_to_check) 
        
        if 'generateContent' not in retrieved_model.supported_generation_methods:
            await interaction.followup.send(f"モデル `{model_name}` は `generateContent` をサポートしていません。タグ付けには利用できません。", ephemeral=True)
            return

        new_model_instance = genai.GenerativeModel(
            retrieved_model.name, 
            safety_settings={ HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                             HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                             HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                             HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,}
        )
        current_gemini_model = retrieved_model.name.replace("models/", "") 
        gemini_model_instance = new_model_instance
        
        save_bot_config({"default_gemini_model": current_gemini_model})
        await interaction.followup.send(f"自動タグ付けのGeminiモデルを `{current_gemini_model}` に設定しました。", ephemeral=True)
        print(f"Geminiモデルが '{current_gemini_model}' に変更されました。 (実行者: {interaction.user})")
    except Exception as e:
        await interaction.followup.send(f"モデル `{model_name}` の設定に失敗しました: {e}", ephemeral=True)
        print(f"Geminiモデル '{model_name}' の設定失敗: {e}")

@gemini_group.command(name="current", description="現在設定されているGeminiモデル名を表示します。(ロール制限あり)")
@is_admin()
async def gemini_current(interaction: discord.Interaction):
    if not gemini_model_instance:
        await interaction.response.send_message(f"Geminiモデルは現在設定されていません、または初期化に失敗しています。", ephemeral=True)
    else:
        await interaction.response.send_message(f"現在設定されている自動タグ付け用Geminiモデルは `{current_gemini_model}` です。", ephemeral=True)

# --- /upload_settings コマンド ---
@upload_settings_group.command(name="set_destination", description="ファイルのアップロード先を設定します。(ロール制限あり)")
@app_commands.describe(destination="アップロード先 ('local' または 'gdrive')")
@app_commands.choices(destination=[
    app_commands.Choice(name="ローカルストレージ", value="local"),
    app_commands.Choice(name="Google Drive", value="gdrive"),
])
@is_admin()
async def set_upload_destination(interaction: discord.Interaction, destination: app_commands.Choice[str]):
    await interaction.response.defer(ephemeral=True)
    new_destination_value = destination.value

    if new_destination_value not in ["local", "gdrive"]:
        await interaction.followup.send("無効なアップロード先です。'local' または 'gdrive' を指定してください。", ephemeral=True)
        return

    if new_destination_value == "gdrive":
        # GDrive関連の設定を確認
        current_gdrive_folder_id = bot_config.get("gdrive_target_folder_id")
        current_gdrive_key_path = bot_config.get("gdrive_service_account_key_path")

        if not current_gdrive_folder_id:
            await interaction.followup.send(
                "Google Drive をアップロード先に設定する前に、`/upload_settings set_gdrive_folder` コマンドでターゲットフォルダIDを設定してください。", 
                ephemeral=True
            )
            return
        if not current_gdrive_key_path or not os.path.exists(current_gdrive_key_path):
             await interaction.followup.send(
                "Google Drive をアップロード先に設定する前に、有効なサービスアカウントキーのパスを `config.json` で設定し、BOTを再起動するか、設定コマンドでキーパスを更新してください。", 
                ephemeral=True
            )
             return

        if not gdrive_service: # サービスが初期化されていない場合
            print("GDrive宛先設定時: GDriveサービスが未初期化のため、再初期化を試みます。")
            initialize_gdrive_service() # 初期化を試みる
            if not gdrive_service: # それでもダメならエラー
                await interaction.followup.send("Google Driveサービスが利用できません。設定（サービスアカウントキー等）を確認してください。", ephemeral=True)
                return
             
    save_bot_config({"upload_destination": new_destination_value})
    # UPLOAD_DESTINATION グローバル変数を更新 (save_bot_config内でも行われるが念のため)
    global UPLOAD_DESTINATION
    UPLOAD_DESTINATION = new_destination_value
    await interaction.followup.send(f"ファイルのアップロード先を「{destination.name}」に設定しました。", ephemeral=True)
    print(f"アップロード先が '{new_destination_value}' に変更されました。(実行者: {interaction.user})")

@upload_settings_group.command(name="set_gdrive_folder", description="Google Driveのアップロード先フォルダIDまたはURLを設定します。(ロール制限あり)")
@app_commands.describe(folder_id_or_url="Google DriveのフォルダID、またはフォルダのURL")
@is_admin()
async def set_gdrive_folder_id(interaction: discord.Interaction, folder_id_or_url: str):
    await interaction.response.defer(ephemeral=True)
    extracted_folder_id = extract_gdrive_folder_id_from_string(folder_id_or_url)

    if not extracted_folder_id or len(extracted_folder_id) < 20: # IDとして短すぎる場合は警告 (一般的なID長に基づく)
        await interaction.followup.send(
            f"設定しようとしているフォルダID「{extracted_folder_id}」は無効な形式のようです。\n"
            "正しいGoogle DriveのフォルダID（通常25文字以上の英数字とハイフン/アンダースコア）、またはフォルダURLを入力してください。", 
            ephemeral=True
        )
        return
        
    save_bot_config({"gdrive_target_folder_id": extracted_folder_id})
    await interaction.followup.send(
        f"Google Driveのアップロード先フォルダIDを `{extracted_folder_id}` に設定しました。\n"
        f"(入力値: `{folder_id_or_url}`)", 
        ephemeral=True
    )
    print(f"GdriveターゲットフォルダIDが '{extracted_folder_id}' に変更されました。(実行者: {interaction.user})")

@upload_settings_group.command(name="current_settings", description="現在のファイルアップロード設定を表示します。(ロール制限あり)")
@is_admin()
async def current_upload_settings(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    
    # bot_configから最新の設定を読み込む
    dest = bot_config.get("upload_destination", DEFAULT_CONFIG["upload_destination"])
    folder_id = bot_config.get("gdrive_target_folder_id") # None もありうる
    create_ym = bot_config.get("gdrive_create_ym_folders", DEFAULT_CONFIG["gdrive_create_ym_folders"])
    gdrive_key_path = bot_config.get("gdrive_service_account_key_path", "未設定")

    embed = discord.Embed(title="現在のアップロード設定", color=discord.Color.blue())
    embed.add_field(name="アップロード先", value=f"`{dest}`", inline=False)
    embed.add_field(name="Google Drive フォルダID", value=f"`{folder_id if folder_id else '未設定'}`", inline=False)
    embed.add_field(name="Google Drive 年月フォルダ作成", value=f"`{create_ym}`", inline=False)
    embed.add_field(name="Google Drive サービスキーパス", value=f"`{gdrive_key_path}`", inline=False)
    
    gdrive_status_msg = "不明"
    if google_drive_libs_available:
        if gdrive_service:
            gdrive_status_msg = "初期化成功"
            # さらに疎通確認を入れるならここ (例: about.get)
            try:
                await execute_gdrive_api_call(gdrive_service.about().get(fields="user").execute)
                gdrive_status_msg += " (API疎通OK)"
            except Exception:
                 gdrive_status_msg += " (API疎通失敗)"
        else:
            gdrive_status_msg = "未初期化または失敗"
            if GDRIVE_SERVICE_ACCOUNT_KEY_PATH and os.path.exists(GDRIVE_SERVICE_ACCOUNT_KEY_PATH):
                 gdrive_status_msg += " (キーファイルは存在)"
            else:
                 gdrive_status_msg += " (キーファイルパス未設定または存在せず)"
    else:
        gdrive_status_msg = "ライブラリ不足"
        
    embed.add_field(name="Google Drive サービス状態", value=gdrive_status_msg, inline=False)
    
    await interaction.followup.send(embed=embed, ephemeral=True)

@bot.tree.command(name="help_nasbot", description="このBOTのコマンド一覧と簡単な説明を表示します。")
async def help_nasbot(interaction: discord.Interaction):
    embed = discord.Embed(title="ファイル管理BOT ヘルプ", description="このBOTで利用可能なコマンド一覧です。", color=discord.Color.blue())
    
    embed.add_field(name="ファイル管理 (`/files`)", value=(
        "`  list [year_month] [keyword]` - 保存されたファイルの一覧を表示します。\n"
        "`  info <filepath>` - 指定されたファイルの詳細情報を表示します。\n" 
        "`  get <filepath>` - 指定されたファイルを取得します。\n"
        "`  delete <filepath>` - 指定されたファイルを削除します。\n"
        "*補足: `filepath` は `YYYYMM/ファイル名` の形式です。オートコンプリートが利用できます。*"
    ), inline=False)
    
    embed.add_field(name="アップロード設定 (`/upload_settings`) (指定ロールのみ)", value=(
        "`  set_destination <local|gdrive>` - アップロード先を設定します。\n"
        "`  set_gdrive_folder <folder_id_or_url>` - Google Driveの保存先フォルダID/URLを設定します。\n"
        "`  current_settings` - 現在のアップロード関連設定を表示します。\n"
    ), inline=False)
    
    embed.add_field(name="Geminiモデル設定 (`/gemini`) (指定ロールのみ)", value=(
        "`  set <model_name>` - 自動タグ付けに使用するGeminiモデルを設定します。\n"
        "`  current` - 現在のGeminiモデル名を表示します。\n"
        "`  list` - 利用可能なGeminiモデルの一覧を表示します。\n"
    ), inline=False)
    
    embed.add_field(name="その他", value=(
        "`/upload_guide` - ファイルのアップロード方法を表示します。\n"
        "`/help_nasbot` - このヘルプを表示します。"
    ), inline=False)
    embed.set_footer(text="ファイルを直接このチャンネルにアップロードすることでも処理が開始されます。")
    await interaction.response.send_message(embed=embed)

# --- コマンドグループをBOTに追加 ---
bot.tree.add_command(gemini_group)
bot.tree.add_command(files_group)
bot.tree.add_command(upload_settings_group)

# --- BOT実行 ---
if __name__ == "__main__":
    if DISCORD_BOT_TOKEN:
        if not GEMINI_API_KEY:
            print("警告: GEMINI_API_KEYが .envファイルに設定されていません。タグ付け機能が制限されます。")
        if not google_drive_libs_available:
            print("警告: Google Drive連携に必要なライブラリが不足しているため、Google Drive関連機能は動作しません。")
        
        bot.run(DISCORD_BOT_TOKEN)
    else:
        print("エラー: DISCORD_BOT_TOKEN が .envファイルに設定されていません。BOTを起動できません。")