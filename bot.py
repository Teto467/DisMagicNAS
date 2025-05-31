import discord
from discord.ext import commands
import os
import datetime
import re
import asyncio # asyncio をインポート
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
    global bot_config, UPLOAD_DESTINATION, GDRIVE_TARGET_FOLDER_ID, GDRIVE_CREATE_YM_FOLDERS, GDRIVE_SERVICE_ACCOUNT_KEY_PATH
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
        bot_config.update(new_settings)
        UPLOAD_DESTINATION = bot_config.get("upload_destination", DEFAULT_CONFIG["upload_destination"])
        GDRIVE_TARGET_FOLDER_ID = bot_config.get("gdrive_target_folder_id")
        GDRIVE_CREATE_YM_FOLDERS = bot_config.get("gdrive_create_ym_folders", DEFAULT_CONFIG["gdrive_create_ym_folders"])
        new_gdrive_key_path = bot_config.get("gdrive_service_account_key_path", DEFAULT_CONFIG["gdrive_service_account_key_path"])
        
        path_changed = (GDRIVE_SERVICE_ACCOUNT_KEY_PATH != new_gdrive_key_path)
        GDRIVE_SERVICE_ACCOUNT_KEY_PATH = new_gdrive_key_path
        # キーパス変更またはフォルダIDが新規設定された場合（かつDriveが有効な場合）は再初期化
        if path_changed or ("gdrive_target_folder_id" in new_settings and new_settings["gdrive_target_folder_id"]):
            if bot_config.get("upload_destination") == "gdrive" or GDRIVE_TARGET_FOLDER_ID: # Drive関連の設定がある場合のみ
                 print("Google Drive関連の設定が変更されたため、サービスを再初期化します。")
                 initialize_gdrive_service()
    except Exception as e: print(f"エラー: '{CONFIG_FILE_NAME}' の保存中に問題が発生しました: {e}")

def extract_gdrive_folder_id_from_string(input_string: str) -> str:
    # 標準的なフォルダURL (e.g., https://drive.google.com/drive/folders/FOLDER_ID_HERE)
    match_folders_url = re.search(r"folders/([a-zA-Z0-9_-]{25,})", input_string)
    if match_folders_url:
        extracted_id = match_folders_url.group(1)
        print(f"URL (folders/) からGoogle DriveフォルダIDを抽出しました: {extracted_id}")
        return extracted_id
    
    # 共有リンクのURL (e.g., https://drive.google.com/drive/u/0/folders/FOLDER_ID_HERE)
    match_shared_url = re.search(r"folders/([a-zA-Z0-9_-]{25,})", input_string) # 上と同じパターンでカバー可能
    if match_shared_url and match_shared_url.group(1) != extracted_id: # 念のため別のIDか確認
        extracted_id_shared = match_shared_url.group(1)
        print(f"URL (shared folders/) からGoogle DriveフォルダIDを抽出しました: {extracted_id_shared}")
        return extracted_id_shared
        
    # URLパラメータからの抽出 (e.g., ?id=FOLDER_ID_HERE, &id=FOLDER_ID_HERE)
    match_id_param = re.search(r"[?&]id=([a-zA-Z0-9_-]{25,})", input_string)
    if match_id_param:
        extracted_id = match_id_param.group(1)
        print(f"URLパラメータからGoogle DriveフォルダIDを抽出しました: {extracted_id}")
        return extracted_id
        
    # それでも見つからなければ、入力文字列が直接IDであると仮定する
    # (ただし、最低限IDとして妥当そうな長さや文字種であるかは別途チェックした方が良いかも)
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
    """ 指定された親フォルダIDの直下にあるサブフォルダの一覧を返す (ページネーション対応) """
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
            
    return sorted(folders_found, key=lambda x: x['name'], reverse=True)

async def list_files_in_gdrive_folder(folder_id: str, service, keyword: str | None = None) -> list[dict]:
    """ 指定されたGoogle DriveのフォルダID内のファイル一覧を返す (ページネーション対応) """
    if not service: return []
    files_found = []

    def _api_call_page(page_token_val=None):
        query = f"mimeType!='application/vnd.google-apps.folder' and trashed=false and '{folder_id}' in parents"
        if keyword: 
            sanitized_keyword = keyword.replace("'", "\\'") 
            query += f" and name contains '{sanitized_keyword}'"
        return service.files().list(q=query,
                                    spaces='drive',
                                    fields='nextPageToken, files(id, name, createdTime, webViewLink, mimeType, size)',
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
            
    return sorted(files_found, key=lambda x: x.get('name'))


def is_admin():
    async def predicate(interaction: discord.Interaction):
        if interaction.guild is None: # DMでは使えない
            await interaction.response.send_message("このコマンドはサーバー内でのみ実行可能です。", ephemeral=True)
            return False
        if not ADMIN_ROLE_NAMES: # 設定に管理者ロールが空の場合
            await interaction.response.send_message("実行に必要なロールがBOTに設定されていません。BOT管理者にお問い合わせください。", ephemeral=True)
            return False
        
        author_roles = [role.name for role in interaction.user.roles]
        if any(admin_role in author_roles for admin_role in ADMIN_ROLE_NAMES):
            return True
        else:
            await interaction.response.send_message(f"このコマンドの実行には、次のいずれかのロールが必要です: `{', '.join(ADMIN_ROLE_NAMES)}`", ephemeral=True)
            return False
    return app_commands.check(predicate)

async def get_tags_from_gemini(file_path, original_filename, mime_type):
    global gemini_model_instance
    if not gemini_model_instance:
        print("Geminiモデルが初期化されていないため、タグ生成をスキップします。")
        return "notags"

    print(f"Gemini APIにファイル '{original_filename}' (MIMEタイプ: {mime_type}) を送信してタグを生成します...")
    uploaded_file_resource = None
    try:
        # ファイルをGemini APIにアップロード
        uploaded_file_resource = genai.upload_file(path=file_path, display_name=original_filename) # TODO: mimetype も指定できるか確認
        print(f"Gemini APIにファイル '{original_filename}' (ID: {uploaded_file_resource.name}) をアップロードしました。")

        prompt = load_tagging_prompt()
        # タグ生成リクエスト
        response = await gemini_model_instance.generate_content_async(
            [prompt, uploaded_file_resource],
            generation_config={"response_mime_type": "text/plain"} # テキスト形式でレスポンスを要求
        )
        
        if response.text.strip() == "タグ抽出不可":
            print("Gemini API: タグ抽出不可と判断されました。")
            return "notags"
            
        tags = response.text.strip()
        # ファイル名に使えない文字などを置換
        sanitized_tags = sanitize_filename_component(tags) # スペースや特定の記号を '_' に置換
        print(f"Gemini APIから取得したタグ: '{sanitized_tags}'")
        return sanitized_tags if sanitized_tags else "notags"

    except Exception as e:
        print(f"Gemini APIでのタグ生成中にエラーが発生しました: {e}")
        return "notags"
    finally:
        # アップロードした一時ファイルをGemini APIから削除
        if uploaded_file_resource and hasattr(uploaded_file_resource, 'name'):
             try:
                 print(f"Gemini APIからアップロードされたファイル '{uploaded_file_resource.name}' の削除を試みます...")
                 genai.delete_file(uploaded_file_resource.name)
                 print(f"Gemini APIからアップロードされたファイル '{uploaded_file_resource.name}' を削除しました。")
             except Exception as e_del:
                 print(f"Gemini APIからアップロードされたファイル {uploaded_file_resource.name} の削除中にエラー: {e_del}")

def get_or_create_drive_folder(parent_folder_id: str, folder_name: str) -> str | None:
    if not gdrive_service or not google_drive_libs_available:
        print("Driveサービスが利用不可のため、フォルダ操作はできません。")
        return None
    try:
        # フォルダを検索
        query = f"mimeType='application/vnd.google-apps.folder' and trashed=false and name='{folder_name}' and '{parent_folder_id}' in parents"
        response = gdrive_service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        folders = response.get('files', [])
        if folders:
            print(f"Driveフォルダ '{folder_name}' が見つかりました (ID: {folders[0].get('id')})。")
            return folders[0].get('id')
        else:
            # フォルダを作成
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
        # 同期的に呼び出すが、initialize_gdrive_service 同様、ボットのメインループ外の操作なので許容範囲か
        # ただし、多数のファイルが同時にアップロードされる場合はここも非同期化検討
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
        
        # ここも execute() はブロッキングコールなので、可能なら非同期化したい
        # uploaded_file = await asyncio.to_thread(
        #     gdrive_service.files().create(body=file_metadata, media_body=media, fields='id, name, webViewLink, thumbnailLink, size').execute
        # )
        # ただし、MediaFileUploadオブジェクトのライフサイクルとの兼ね合いで単純なto_threadでは難しい場合がある
        # google-api-python-client の非同期アップロード方法を調査する必要があるかもしれない
        # 現状は同期的なままとしておく
        uploaded_file = gdrive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, name, webViewLink, thumbnailLink, size' # thumbnailLinkは画像・動画の場合のみ
        ).execute()

        print(f"ファイル '{uploaded_file.get('name')}' がGoogle Driveにアップロードされました。ID: {uploaded_file.get('id')}, Link: {uploaded_file.get('webViewLink')}")
        return uploaded_file
    except Exception as e:
        print(f"Google Driveへのファイルアップロード中にエラーが発生しました: {e}")
        return None

class ConfirmDeleteView(discord.ui.View):
    def __init__(self, author_id: int, file_path_to_delete: str, filename_display: str):
        super().__init__(timeout=30.0) # タイムアウトを30秒に設定
        self.author_id = author_id
        self.file_path_to_delete = file_path_to_delete
        self.filename_display = filename_display
        self.confirmed: bool | None = None # ユーザーの選択状態
        self.interaction_message: discord.InteractionMessage | None = None # インタラクションメッセージを保持

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        # コマンド実行者本人のみ操作可能
        if interaction.user.id != self.author_id:
            await interaction.response.send_message("この操作はコマンドを実行した本人のみが行えます。", ephemeral=True)
            return False
        return True

    @discord.ui.button(label="削除実行", style=discord.ButtonStyle.danger)
    async def confirm_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.confirmed = True
        # 全てのボタンを無効化
        for item in self.children:
            item.disabled = True
        await interaction.response.edit_message(content=f"ファイル `{self.filename_display}` の削除処理を開始します...", view=self)
        self.stop() # Viewの待機を停止

    @discord.ui.button(label="キャンセル", style=discord.ButtonStyle.secondary)
    async def cancel_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.confirmed = False
        for item in self.children:
            item.disabled = True
        await interaction.response.edit_message(content=f"ファイル `{self.filename_display}` の削除はキャンセルされました。", view=self)
        self.stop()

    async def on_timeout(self):
        if self.confirmed is None: # ユーザーが何も選択しなかった場合
            for item in self.children:
                item.disabled = True
            if self.interaction_message:
                try:
                    await self.interaction_message.edit(content=f"ファイル `{self.filename_display}` の削除確認がタイムアウトしました。", view=self)
                except discord.NotFound: # メッセージが既に削除されている場合など
                    pass
                except discord.HTTPException as e:
                    print(f"タイムアウト時のメッセージ編集エラー: {e}")
            self.stop()

@bot.event
async def on_ready():
    global current_gemini_model
    print(f'{bot.user.name} としてログインしました (ID: {bot.user.id})')
    print(f'監視中のサーバー数: {len(bot.guilds)}')
    print(f'ベースアップロードフォルダ(ローカル): {os.path.abspath(BASE_UPLOAD_FOLDER)}')
    print(f'現在のアップロード先: {UPLOAD_DESTINATION}') # グローバル変数 UPLOAD_DESTINATION を参照
    print(f'Geminiコマンド管理者ロール: {ADMIN_ROLE_NAMES}')
    if gemini_model_instance:
        print(f'使用中Geminiモデル: {current_gemini_model}')
    else:
        print('Geminiモデルは初期化されていません。')
    
    load_tagging_prompt() # タグ付けプロンプトを読み込む

    # ベースアップロードフォルダが存在しなければ作成 (アップロード先に関わらずtemp等で使う可能性)
    if not os.path.exists(BASE_UPLOAD_FOLDER):
        os.makedirs(BASE_UPLOAD_FOLDER)
        print(f"ベースフォルダ '{BASE_UPLOAD_FOLDER}' を作成しました。")

    initialize_gdrive_service() # Google Driveサービスを初期化

    try:
        await bot.tree.sync() # スラッシュコマンドを同期
        print("スラッシュコマンドを同期しました。")
    except Exception as e:
        print(f"スラッシュコマンドの同期に失敗しました: {e}")
    print('------')

@bot.event
async def on_message(message):
    if message.author == bot.user: return # ボット自身のメッセージは無視
    if message.attachments: # 添付ファイルがある場合
        ctx = await bot.get_context(message) # コンテキストを取得 (サーバー情報などのため)
        for attachment in message.attachments:
            # 対応するファイル形式 (画像と動画)
            allowed_image_types = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp')
            allowed_video_types = ('.mp4', '.mov', '.avi', '.mkv', '.webm') # 主要な動画形式
            
            file_ext = os.path.splitext(attachment.filename)[1].lower()
            if not (file_ext in allowed_image_types or file_ext in allowed_video_types):
                await message.channel.send(f"ファイル '{attachment.filename}' の形式 ({file_ext}) はサポートされていません。画像または動画ファイルをアップロードしてください。")
                continue

            # ファイルサイズ制限 (Discordのサーバーごとの制限を考慮)
            limit_bytes = 8 * 1024 * 1024 # デフォルト8MB (DMやNitroなしの基準)
            if ctx.guild and hasattr(ctx.guild, 'filesize_limit'): # サーバーコンテキストで、かつ属性が存在すれば
                limit_bytes = ctx.guild.filesize_limit
            
            if attachment.size > limit_bytes:
                 await message.channel.send(f"ファイル '{attachment.filename}' ({attachment.size // 1024 // 1024}MB) はサイズが大きすぎます (サーバー上限: {limit_bytes // 1024 // 1024}MB)。")
                 continue

            # 一時保存先 (BASE_UPLOAD_FOLDER直下のtempフォルダ)
            # BASE_UPLOAD_FOLDERはon_readyで作成されることを期待
            temp_dir = os.path.join(BASE_UPLOAD_FOLDER, "temp")
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            
            # 一時ファイル名 (衝突を避けるため attachment.id を含める)
            temp_save_path = os.path.join(temp_dir, f"temp_{attachment.id}_{sanitize_filename_component(attachment.filename)}")
            
            await attachment.save(temp_save_path) # ファイルを一時保存
            processing_msg = await message.channel.send(f"ファイル '{attachment.filename}' を処理中... 自動タグ付けを開始します。")

            tags_str = "notags"
            if gemini_model_instance:
                try:
                    # 画像の場合、Pillowで有効性を軽くチェック (破損ファイル対策)
                    if file_ext in allowed_image_types:
                        try:
                            img = Image.open(temp_save_path)
                            img.verify() # ヘッダーチェックなど
                            img.close()
                        except Exception as img_err:
                            await processing_msg.edit(content=f"ファイル '{attachment.filename}' は有効な画像ではないようです。処理を中断します。({img_err})")
                            if os.path.exists(temp_save_path): os.remove(temp_save_path)
                            continue
                    
                    # Geminiでタグ生成 (ファイルパス、元のファイル名、MIMEタイプを渡す)
                    tags_str = await get_tags_from_gemini(temp_save_path, attachment.filename, attachment.content_type)
                except Exception as e:
                    print(f"タグ付け処理中にエラー: {e}")
                    await processing_msg.edit(content=f"ファイル '{attachment.filename}' のタグ付け中にエラーが発生しました。タグなしで処理を続行します。")
                    tags_str = "notags" # エラー時はタグなし
            else:
                await processing_msg.edit(content=f"ファイル '{attachment.filename}' を処理中... (Gemini API未設定のためタグ付けスキップ)")

            # 新しいファイル名の生成 (日付_タグ_元ファイル名.拡張子)
            date_str = datetime.datetime.now().strftime("%Y%m%d")
            original_filename_no_ext, original_ext = os.path.splitext(attachment.filename)
            sanitized_original_filename = sanitize_filename_component(original_filename_no_ext)
            new_filename = f"{date_str}_{tags_str}_{sanitized_original_filename}{original_ext}"
            
            display_tags_on_message = tags_str.replace("_", "-") if tags_str != "notags" else "なし"

            current_upload_dest_on_message = bot_config.get("upload_destination", "local") # 現在の設定を再取得
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
                # Google Driveアップロード後は一時ファイルを削除
                if os.path.exists(temp_save_path):
                    try: os.remove(temp_save_path); print(f"一時ファイル '{temp_save_path}' を削除しました。")
                    except Exception as e_rm: print(f"一時ファイル '{temp_save_path}' の削除失敗: {e_rm}")

            elif current_upload_dest_on_message == "local":
                local_ym_folder = create_year_month_folder_if_not_exists(BASE_UPLOAD_FOLDER)
                final_save_path = os.path.join(local_ym_folder, new_filename)
                try:
                    os.rename(temp_save_path, final_save_path) # 一時ファイルを最終保存場所に移動
                    print(f"ファイル '{attachment.filename}' を '{final_save_path}' に保存しました。")
                    await processing_msg.edit(content=(
                        f"ファイル '{attachment.filename}' をローカルに保存しました: '{new_filename}'\n自動タグ: `{display_tags_on_message}`"
                    ))
                except Exception as e:
                    print(f"ローカル保存エラー: {e}")
                    await processing_msg.edit(content=f"'{attachment.filename}' のローカル保存中にエラーが発生しました。")
                    if os.path.exists(temp_save_path): # 移動失敗時は一時ファイルを削除
                        try: os.remove(temp_save_path); print(f"エラー発生のため一時ファイル '{temp_save_path}' を削除しました。")
                        except Exception as e_rm: print(f"一時ファイル '{temp_save_path}' の削除失敗: {e_rm}")
            else:
                print(f"不明なアップロード先が設定されています: {current_upload_dest_on_message}")
                await processing_msg.edit(content=f"アップロード先の設定が不明なため、'{attachment.filename}' の処理を中断しました。")
                if os.path.exists(temp_save_path): # 不明な場合も一時ファイル削除
                    try: os.remove(temp_save_path); print(f"不明なアップロード先のため一時ファイル '{temp_save_path}' を削除しました。")
                    except Exception as e_rm: print(f"一時ファイル '{temp_save_path}' の削除失敗: {e_rm}")

    await bot.process_commands(message) # 通常のコマンド処理も行う

# --- オートコンプリート用の関数 ---
async def gemini_model_autocomplete(interaction: discord.Interaction, current: str) -> list[app_commands.Choice[str]]:
    choices = []
    if not GEMINI_API_KEY or not genai: return [] # APIキーがないかライブラリがなければ空
    try:
        for model in genai.list_models():
            # generateContent をサポートするモデルのみをリストアップ
            if 'generateContent' in model.supported_generation_methods:
                model_display_name = model.name.replace("models/", "") # "models/" プレフィックスを除去
                if current.lower() in model_display_name.lower(): # 入力された文字でフィルタリング
                    choice_name = f"{model_display_name} ({model.display_name})"
                    if len(choice_name) > 100: choice_name = model_display_name[:97] + "..." # 文字数制限対策
                    choices.append(app_commands.Choice(name=choice_name, value=model_display_name))
            if len(choices) >= 25: break # Discordの候補数上限
    except Exception as e:
        print(f"Geminiモデルのオートコンプリート中にエラー: {e}")
        return []
    return choices

async def year_month_autocomplete(interaction: discord.Interaction, current: str) -> list[app_commands.Choice[str]]:
    # TODO: GDrive対応時は、current_upload_dest をみて GDrive からも候補を取得する
    choices = []
    ym_folders = set()
    if not os.path.exists(BASE_UPLOAD_FOLDER): return [] # ベースフォルダがなければ空
    try:
        for item in os.listdir(BASE_UPLOAD_FOLDER):
            if os.path.isdir(os.path.join(BASE_UPLOAD_FOLDER, item)) and len(item) == 6 and item.isdigit():
                ym_folders.add(item)
        
        for folder_name in sorted(list(ym_folders), reverse=True): # 新しい順に
            if current.lower() in folder_name.lower():
                choices.append(app_commands.Choice(name=folder_name, value=folder_name))
            if len(choices) >= 25: break
    except Exception as e:
        print(f"year_month_autocomplete 中にエラー: {e}")
        return []
    return choices

async def filename_autocomplete(interaction: discord.Interaction, current: str) -> list[app_commands.Choice[str]]:
    # TODO: GDrive対応時は、current_upload_dest をみて GDrive からも候補を取得する
    choices = []
    specific_ym_folder = None # 特定のYYYYMMフォルダが指定されているか
    current_filename_part_to_search = current

    if not os.path.exists(BASE_UPLOAD_FOLDER): return []

    # "YYYYMM/filename_part" のような入力をパース
    if '/' in current and len(current.split('/')[0]) == 6 and current.split('/')[0].isdigit():
        parts = current.split('/', 1)
        ym_part = parts[0]
        path_to_check = os.path.join(BASE_UPLOAD_FOLDER, ym_part)
        if os.path.isdir(path_to_check):
            specific_ym_folder = path_to_check
            current_filename_part_to_search = parts[1] if len(parts) > 1 else "" # YYYYMM/ の場合は全ファイル

    # 検索対象のフォルダパスリストを作成
    folders_to_search = []
    if specific_ym_folder:
        folders_to_search.append(specific_ym_folder)
    else: # YYYYMMの指定がなければ、全てのYYYYMMフォルダを検索
        for item in sorted(os.listdir(BASE_UPLOAD_FOLDER), reverse=True): # 新しい順
            item_path = os.path.join(BASE_UPLOAD_FOLDER, item)
            if os.path.isdir(item_path) and len(item) == 6 and item.isdigit():
                folders_to_search.append(item_path)
    
    for folder_path in folders_to_search:
        year_month_dir = os.path.basename(folder_path) # "YYYYMM"
        try:
            for fname in sorted(os.listdir(folder_path)): # ファイル名順
                if os.path.isfile(os.path.join(folder_path, fname)):
                    if current_filename_part_to_search.lower() in fname.lower():
                        # 選択肢の表示名と値を設定
                        suffix = f" (in {year_month_dir})"
                        allowed_fname_len = 100 - len(suffix) # Discordの表示名上限対策
                        if allowed_fname_len < 1: display_fname = "" # 極端に短い場合
                        elif len(fname) <= allowed_fname_len: display_fname = fname
                        else: display_fname = fname[:max(0, allowed_fname_len - 3)] + "..."
                        
                        final_choice_name = f"{display_fname}{suffix}"
                        if len(final_choice_name) > 100: # 再度チェック
                            final_choice_name = final_choice_name[:97] + "..."
                            
                        choices.append(app_commands.Choice(name=final_choice_name, value=f"{year_month_dir}/{fname}"))
                        if len(choices) >= 25: break
        except Exception as e:
            print(f"filename_autocomplete でフォルダ '{folder_path}' のスキャン中にエラー: {e}")
        if len(choices) >= 25: break # 外側のループも抜ける
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
@app_commands.autocomplete(year_month=year_month_autocomplete) # TODO: このオートコンプリートもGDRIVE対応が必要
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
                await interaction.followup.send(f"指定された年月フォルダ '{year_month}' は見つかりません。")
                return
        else:
            if os.path.exists(BASE_UPLOAD_FOLDER):
                for item in sorted(os.listdir(BASE_UPLOAD_FOLDER), reverse=True): 
                    item_path = os.path.join(BASE_UPLOAD_FOLDER, item)
                    if os.path.isdir(item_path) and len(item) == 6 and item.isdigit():
                        search_paths.append(item_path)
            
        if not search_paths:
            msg = "検索対象のフォルダが見つかりません。"
            if year_month: msg = f"指定された年月フォルダ '{year_month}' は見つかりません。"
            elif not os.path.exists(BASE_UPLOAD_FOLDER): msg = f"ベースアップロードフォルダ '{BASE_UPLOAD_FOLDER}' が見つかりません。"
            await interaction.followup.send(msg)
            return

        for folder_to_scan in search_paths:
            try:
                current_year_month = os.path.basename(folder_to_scan)
                for fname in sorted(os.listdir(folder_to_scan)):
                    if os.path.isfile(os.path.join(folder_to_scan, fname)):
                        if keyword and keyword.lower() not in fname.lower():
                            continue
                        parsed_info = parse_bot_filename(fname)
                        found_files_details.append({
                            "fullname": fname,
                            "date": parsed_info["date"],
                            "tags": parsed_info["tags_display"],
                            "original_name": parsed_info["original_stem"],
                            "year_month": current_year_month
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
                        "fullname": gfile_name,
                        "date": parsed_info["date"], 
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
@app_commands.autocomplete(filepath=filename_autocomplete) # TODO: GDrive対応
async def files_info(interaction: discord.Interaction, filepath: str):
    # TODO: GDrive対応。current_upload_dest をみて処理を分岐する
    await interaction.response.defer()
    try:
        ym_dir, filename = filepath.split('/', 1)
    except ValueError:
        await interaction.followup.send("ファイルパスの形式が正しくありません。YYYYMM/ファイル名の形式で入力してください。", ephemeral=True)
        return

    full_path = os.path.join(BASE_UPLOAD_FOLDER, ym_dir, filename)
    if not os.path.exists(full_path) or not os.path.isfile(full_path):
        await interaction.followup.send(f"ファイル `{filepath}` がローカルに見つかりません。") # GDrive未対応のためローカルのみのメッセージ
        return
    
    try:
        parsed_info = parse_bot_filename(filename)
        file_size_bytes = os.path.getsize(full_path)
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
        except Exception as e_time:
            print(f"最終更新日時の取得エラー: {e_time}")
        
        await interaction.followup.send(embed=embed)
    except Exception as e:
        print(f"/files info処理中にエラー: {e}")
        await interaction.followup.send(f"ファイル情報の取得中にエラーが発生しました: {e}")

@files_group.command(name="delete", description="指定された保存済みファイルをサーバーから削除します。")
@app_commands.describe(filepath="削除するファイル (年月フォルダ/ファイル名)")
@app_commands.autocomplete(filepath=filename_autocomplete) # TODO: GDrive対応
async def files_delete(interaction: discord.Interaction, filepath: str):
    # TODO: GDrive対応。current_upload_dest をみて処理を分岐する
    await interaction.response.defer() # deferを先に行う
    try:
        ym_dir, filename = filepath.split('/', 1)
    except ValueError:
        await interaction.followup.send("ファイルパスの形式が正しくありません。YYYYMM/ファイル名の形式で入力してください。", ephemeral=True)
        return

    full_path = os.path.join(BASE_UPLOAD_FOLDER, ym_dir, filename)
    if not os.path.exists(full_path) or not os.path.isfile(full_path):
        await interaction.followup.send(f"ファイル `{filepath}` がローカルに見つかりません。") # GDrive未対応
        return

    view = ConfirmDeleteView(author_id=interaction.user.id, file_path_to_delete=full_path, filename_display=filename)
    interaction_message = await interaction.followup.send(
        f"**警告:** ファイル `{filename}` を本当に削除しますか？この操作は取り消せません。(実行者: {interaction.user.mention})", 
        view=view
    )
    view.interaction_message = interaction_message # viewにメッセージオブジェクトを渡す
    
    await view.wait() # ユーザーの応答を待つ

    if view.confirmed is True:
        try:
            os.remove(full_path)
            print(f"ユーザー {interaction.user} によってファイル {full_path} が削除されました。")
            await interaction_message.edit(content=f"ファイル `{filename}` を削除しました。(実行者: {interaction.user.mention})", view=None)
        except Exception as e:
            print(f"ファイル削除エラー ({full_path}): {e}")
            await interaction_message.edit(content=f"ファイル `{filename}` の削除中にエラーが発生しました: {e}", view=None)
    # キャンセルまたはタイムアウトの場合は、view側でメッセージが編集される

@files_group.command(name="get", description="指定された保存済みファイルを取得します。")
@app_commands.describe(filepath="取得するファイル (年月フォルダ/ファイル名)")
@app_commands.autocomplete(filepath=filename_autocomplete) # TODO: GDrive対応  <- "app_app_commands" を "app_commands" に修正
async def files_get(interaction: discord.Interaction, filepath: str):
    # TODO: GDrive対応。current_upload_dest をみて処理を分岐する
    await interaction.response.defer() # deferを先に行う
    try:
        ym_dir, filename = filepath.split('/', 1)
    except ValueError:
        await interaction.followup.send("ファイルパスの形式が正しくありません。YYYYMM/ファイル名の形式で入力してください。", ephemeral=True)
        return

    full_path = os.path.join(BASE_UPLOAD_FOLDER, ym_dir, filename)
    if not os.path.exists(full_path) or not os.path.isfile(full_path):
        await interaction.followup.send(f"ファイル `{filepath}` がローカルに見つかりません。") # GDrive未対応
        return

    # Discordのファイルサイズ制限を確認
    limit_bytes = 8 * 1024 * 1024 # デフォルト8MB
    if interaction.guild: # サーバー内であればサーバーの制限値を使用
        limit_bytes = interaction.guild.filesize_limit
    
    file_size_bytes = os.path.getsize(full_path)
    if file_size_bytes > limit_bytes:
        await interaction.followup.send(
            f"ファイル `{filename}` ({round(file_size_bytes / (1024*1024), 2)} MB) はDiscordの送信サイズ上限を超えています "
            f"(上限: {round(limit_bytes / (1024*1024), 2)} MB)。"
        )
        return
        
    try:
        discord_file = discord.File(full_path, filename=filename)
        await interaction.followup.send(f"ファイル `{filename}` を送信します: (要求者: {interaction.user.mention})", file=discord_file)
    except Exception as e:
        print(f"ファイル送信エラー ({full_path}): {e}")
        await interaction.followup.send(f"ファイル `{filename}` の送信中にエラーが発生しました: {e}")

# --- /gemini サブコマンド ---
@gemini_group.command(name="list", description="利用可能なGeminiモデルの一覧を表示します。(ロール制限あり)")
@is_admin()
async def gemini_list(interaction: discord.Interaction):
    if not GEMINI_API_KEY:
        await interaction.response.send_message("Gemini APIキーが設定されていません。", ephemeral=True)
        return
    if not genai: # genaiモジュールがロードできていない場合
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
                # メッセージ長がDiscordの制限を超えないように分割送信
                if len("".join(models_info_parts)) + len(current_part) > 1900: # 2000字制限のマージン
                    await interaction.followup.send("".join(models_info_parts), ephemeral=True)
                    models_info_parts = [current_part] # 新しいメッセージを開始
                else:
                    models_info_parts.append(current_part)
                count += 1
        
        if count == 0 and len(models_info_parts) == 1 and models_info_parts[0].endswith(":\n"):
            # 最初の "利用可能な..." しかない場合
             models_info_parts.append("利用可能なGeminiモデルが見つかりませんでした。")

        if models_info_parts: # 残りのメッセージがあれば送信
            final_message = "".join(models_info_parts)
            if final_message.strip() and not (count == 0 and final_message.endswith(":\n") and len(final_message.splitlines()) ==1) : # 空でない、かつ初期メッセージのみでない
                 await interaction.followup.send(final_message, ephemeral=True)
            elif count == 0 : # generateContentサポートモデルが一つもなかった場合
                await interaction.followup.send("利用可能なGeminiモデル (generateContentサポート) が見つかりませんでした。",ephemeral=True)

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
        # モデル名が "models/" で始まっていなければ付加 (genai.get_modelの仕様に合わせる)
        full_model_name_to_check = model_name if model_name.startswith("models/") else f"models/{model_name}"
        
        retrieved_model = genai.get_model(full_model_name_to_check) # モデルの存在確認と情報取得
        
        # generateContentをサポートしているか確認
        if 'generateContent' not in retrieved_model.supported_generation_methods:
            await interaction.followup.send(f"モデル `{model_name}` は `generateContent` をサポートしていません。タグ付けには利用できません。", ephemeral=True)
            return

        # 新しいモデルインスタンスを作成
        new_model_instance = genai.GenerativeModel(
            retrieved_model.name, # APIから取得した正式なモデル名を使用
            safety_settings={ HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                             HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                             HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                             HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,}
        )
        current_gemini_model = retrieved_model.name.replace("models/", "") # "models/"を除いた名前を保持
        gemini_model_instance = new_model_instance
        
        # 設定ファイルにも保存 (デフォルトモデルとして)
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

    if new_destination_value not in ["local", "gdrive"]: # 基本的にchoicesで制限されるはずだが念のため
        await interaction.followup.send("無効なアップロード先です。'local' または 'gdrive' を指定してください。", ephemeral=True)
        return

    if new_destination_value == "gdrive":
        if not GDRIVE_TARGET_FOLDER_ID: # DriveのフォルダIDが未設定の場合
            await interaction.followup.send(
                "Google Drive をアップロード先に設定する前に、`/upload_settings set_gdrive_folder` コマンドでターゲットフォルダIDを設定してください。", 
                ephemeral=True
            )
            return
        if not gdrive_service and google_drive_libs_available: # GDriveサービスが未初期化だがライブラリはある場合
            initialize_gdrive_service() # 初期化を試みる
        if not gdrive_service: # それでもダメならエラー
             await interaction.followup.send("Google Driveサービスが利用できません。設定（サービスアカウントキー等）を確認してください。", ephemeral=True)
             return
             
    save_bot_config({"upload_destination": new_destination_value})
    await interaction.followup.send(f"ファイルのアップロード先を「{destination.name}」に設定しました。", ephemeral=True)
    print(f"アップロード先が '{new_destination_value}' に変更されました。(実行者: {interaction.user})")

@upload_settings_group.command(name="set_gdrive_folder", description="Google Driveのアップロード先フォルダIDまたはURLを設定します。(ロール制限あり)")
@app_commands.describe(folder_id_or_url="Google DriveのフォルダID、またはフォルダのURL")
@is_admin()
async def set_gdrive_folder_id(interaction: discord.Interaction, folder_id_or_url: str):
    await interaction.response.defer(ephemeral=True)
    extracted_folder_id = extract_gdrive_folder_id_from_string(folder_id_or_url)

    if not extracted_folder_id or len(extracted_folder_id) < 20: # IDとして短すぎる場合は警告
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
    folder_id = bot_config.get("gdrive_target_folder_id", "未設定")
    create_ym = bot_config.get("gdrive_create_ym_folders", DEFAULT_CONFIG["gdrive_create_ym_folders"])
    gdrive_key_path = bot_config.get("gdrive_service_account_key_path", "未設定")

    embed = discord.Embed(title="現在のアップロード設定", color=discord.Color.blue())
    embed.add_field(name="アップロード先", value=f"`{dest}`", inline=False)
    embed.add_field(name="Google Drive フォルダID", value=f"`{folder_id if folder_id else '未設定'}`", inline=False) # Noneの場合も考慮
    embed.add_field(name="Google Drive 年月フォルダ作成", value=f"`{create_ym}`", inline=False)
    embed.add_field(name="Google Drive サービスキーパス", value=f"`{gdrive_key_path}`", inline=False)
    
    gdrive_status = "初期化成功" if gdrive_service else \
                    ("未初期化または失敗 (ライブラリあり)" if google_drive_libs_available else "ライブラリ不足")
    embed.add_field(name="Google Drive サービス状態", value=gdrive_status, inline=False)
    
    await interaction.followup.send(embed=embed, ephemeral=True)

@bot.tree.command(name="help_nasbot", description="このBOTのコマンド一覧と簡単な説明を表示します。")
async def help_nasbot(interaction: discord.Interaction):
    embed = discord.Embed(title="ファイル管理BOT ヘルプ", description="このBOTで利用可能なコマンド一覧です。", color=discord.Color.blue())
    
    embed.add_field(name="ファイル管理 (`/files`)", value=(
        "`  list [year_month] [keyword]` - 保存されたファイルの一覧を表示します。\n"
        "`  info <filepath>` - 指定されたファイルの詳細情報を表示します。\n" # TODO: GDrive対応時にfilepathの形式も考慮
        "`  get <filepath>` - 指定されたファイルを取得します。\n" # TODO: GDrive対応
        "`  delete <filepath>` - 指定されたファイルを削除します。\n" # TODO: GDrive対応
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