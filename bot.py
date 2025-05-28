import discord
from discord.ext import commands
import os
import datetime
import re
import asyncio
import json # [cite: 7]
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold # Gemini APIのコンテンツフィルター設定用
from PIL import Image # 画像のバリデーションや前処理用

# --- 設定 ---
# .envファイルから環境変数を読み込む
load_dotenv()
DISCORD_BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
BASE_UPLOAD_FOLDER = os.getenv('BASE_UPLOAD_FOLDER', 'uploads') # .envになければ 'uploads' をデフォルトに
ADMIN_ROLE_NAMES_STR = os.getenv('ADMIN_ROLE_NAMES', 'BOT管理者,運営スタッフ') # カンマ区切りのロール名文字列
ADMIN_ROLE_NAMES = [name.strip() for name in ADMIN_ROLE_NAMES_STR.split(',')]
DEFAULT_GEMINI_MODEL = os.getenv('DEFAULT_GEMINI_MODEL', 'gemini-1.5-flash-latest') # .envになければ指定モデル

# --- Gemini API 初期化 ---
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY) # [cite: 6]
    # 使用するモデルを格納する変数（コマンドで変更可能にするため）
    current_gemini_model = DEFAULT_GEMINI_MODEL
    # Gemini APIのクライアント (generation_config はここでグローバルに設定も可能)
    gemini_model_instance = genai.GenerativeModel(
        current_gemini_model,
        # 安全性設定: 全てのカテゴリでブロック閾値を「なし」に設定 (デモ用、本番では要検討)
        safety_settings={
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )
else:
    print("エラー: GEMINI_API_KEYが設定されていません。 .envファイルを確認してください。")
    exit() # APIキーがない場合は終了

# --- BOT 初期化 ---
intents = discord.Intents.default()
intents.message_content = True
intents.members = True # ロール情報を取得するために必要
bot = commands.Bot(command_prefix='/', intents=intents) # コマンドプレフィックスを '/' に設定

# --- ヘルパー関数 ---
def sanitize_filename_component(text):
    """ファイル名やタグに使用できない文字をアンダースコアに置換する"""
    # OSのファイル名制限も考慮し、長すぎる場合もケアが必要だが、ここでは文字置換のみ
    return re.sub(r'[\\/*?:"<>|\s]', '_', text) # スペースもアンダースコアに置換

def get_file_icon(extension):
    """拡張子に基づいて絵文字アイコンを返す（簡易版）"""
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
    """ベースフォルダ配下に YYYYMM 形式のフォルダを作成する"""
    now = datetime.datetime.now()
    year_month_folder_name = now.strftime("%Y%m")
    year_month_folder_path = os.path.join(base_folder, year_month_folder_name)
    if not os.path.exists(year_month_folder_path):
        os.makedirs(year_month_folder_path)
        print(f"年月フォルダ '{year_month_folder_path}' を作成しました。")
    return year_month_folder_path

# --- 管理者チェック ---
def is_admin():
    """コマンド実行者が管理者ロールを持っているかチェックするデコレータ"""
    async def predicate(ctx):
        if ctx.guild is None: # DMではロールチェック不可
            await ctx.send("このコマンドはサーバー内でのみ実行可能です。")
            return False
        # コマンド実行者のロール名を取得
        author_roles = [role.name for role in ctx.author.roles]
        # ADMIN_ROLE_NAMES のいずれかのロールを持っていればTrue
        if any(admin_role in author_roles for admin_role in ADMIN_ROLE_NAMES):
            return True
        await ctx.send("このコマンドを実行する権限がありません。")
        return False
    return commands.check(predicate)

# --- Gemini API 関連 ---
async def get_tags_from_gemini(file_path, original_filename, mime_type):
    """Gemini APIを使用してファイルの内容からタグを生成する"""
    print(f"Gemini APIにファイル '{original_filename}' (MIMEタイプ: {mime_type}) を送信してタグを生成します...")
    try:
        # 画像と動画で処理を分ける (動画はまだSDKが直接対応していない場合があるため、ここでは画像のみを対象とする)
        # ドキュメントには動画理解の記述があるため、SDKの進化により対応可能になる想定 [cite: 1]
        # 現状の `google-generativeai` SDK (genai.upload_file) はローカルファイルパスを直接サポート
        uploaded_file = genai.upload_file(path=file_path, display_name=original_filename) # mime_typeは自動検出されることが多い

        prompt = (
            "このファイルの内容を詳細に分析し、関連性の高いキーワードを5つ提案してください。"
            "各キーワードは簡潔な日本語で、ハイフン(-)で連結可能な形式でお願いします。"
            "例: 風景-自然-山-川-晴天"
            "もし内容が不明瞭な場合やキーワード抽出が難しい場合は、'タグ抽出不可'とだけ返してください。"
        )
        # モデルインスタンスを使ってコンテンツを生成
        response = gemini_model_instance.generate_content(
            [prompt, uploaded_file],
            generation_config={"response_mime_type": "text/plain"} # タグなのでプレーンテキストで十分
        )

        if response.text.strip() == "タグ抽出不可":
            print("Gemini API: タグ抽出不可と判断されました。")
            return "notags"

        # 生成されたテキストからタグを抽出（例： "タグ1-タグ2-タグ3" のような形式を期待）
        tags = response.text.strip()
        # サニタイズ（Geminiが禁止文字を返す可能性も考慮）
        sanitized_tags = sanitize_filename_component(tags) # ハイフンは許可し、他の禁止文字を置換
        print(f"Gemini APIから取得したタグ: '{sanitized_tags}'")
        return sanitized_tags if sanitized_tags else "notags"

    except Exception as e:
        print(f"Gemini APIでのタグ生成中にエラーが発生しました: {e}")
        return "notags" # エラー時は "notags"
    finally:
        # アップロードしたファイルは、タグ付け後に不要であれば削除を検討
        # genai.delete_file(uploaded_file.name) # 必要に応じてコメントアウト解除
        pass


# --- イベントハンドラ ---
@bot.event
async def on_ready():
    print(f'{bot.user.name} としてログインしました (ID: {bot.user.id})')
    print(f'監視中のサーバー数: {len(bot.guilds)}')
    print(f'ベースアップロードフォルダ: {os.path.abspath(BASE_UPLOAD_FOLDER)}')
    print(f'管理者ロール: {ADMIN_ROLE_NAMES}')
    print(f'使用中Geminiモデル: {current_gemini_model}')
    if not os.path.exists(BASE_UPLOAD_FOLDER):
        os.makedirs(BASE_UPLOAD_FOLDER)
        print(f"ベースフォルダ '{BASE_UPLOAD_FOLDER}' を作成しました。")
    try:
        await bot.tree.sync() # スラッシュコマンドを同期
        print("スラッシュコマンドを同期しました。")
    except Exception as e:
        print(f"スラッシュコマンドの同期に失敗しました: {e}")
    print('------')

@bot.event
async def on_message(message):
    """メッセージ受信時の処理。主にファイルアップロードを処理。"""
    if message.author == bot.user:
        return

    # ファイルが添付されている場合
    if message.attachments:
        # まず年月フォルダを準備
        year_month_folder_path = create_year_month_folder_if_not_exists(BASE_UPLOAD_FOLDER)
        ctx = await bot.get_context(message) # コマンド実行コンテキストを取得するため

        for attachment in message.attachments:
            # 対応ファイル形式チェック (要件定義書 3.1)
            allowed_image_types = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp')
            allowed_video_types = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
            file_ext = os.path.splitext(attachment.filename)[1].lower()

            if not (file_ext in allowed_image_types or file_ext in allowed_video_types):
                await message.channel.send(
                    f"ファイル '{attachment.filename}' の形式 ({file_ext}) はサポートされていません。\n"
                    f"対応形式 (画像): {', '.join(allowed_image_types)}\n"
                    f"対応形式 (動画): {', '.join(allowed_video_types)}"
                )
                continue # 次の添付ファイルへ

            # Discordのファイルサイズ制限に注意 (要件定義書 3.1)
            # discord.pyでは attachment.size でバイト単位のサイズが取れる
            if attachment.size > 8 * 1024 * 1024 and not (ctx.guild and ctx.guild.premium_tier >= 1): # 8MB超でブーストなしの場合
                 # Nitroユーザーの制限はより複雑なので、ここでは簡易的に8MBで線引き
                await message.channel.send(
                    f"ファイル '{attachment.filename}' ({attachment.size // 1024 // 1024}MB) はサイズが大きすぎます。"
                    "サーバーブーストレベルに応じて上限が緩和されますが、基本は8MBまでです。"
                )
                continue

            temp_save_path = os.path.join(year_month_folder_path, f"temp_{attachment.filename}")
            await attachment.save(temp_save_path)
            processing_msg = await message.channel.send(f"ファイル '{attachment.filename}' をアップロード中... 自動タグ付け処理を開始します。しばらくお待ちください。")

            # Gemini APIでタグ付け (非同期で実行し、完了を待つ)
            tags_str = "notags"
            try:
                # 画像の場合はPillowで一度開いてみる（破損チェックや形式確認のため）
                if file_ext in allowed_image_types:
                    try:
                        img = Image.open(temp_save_path)
                        img.verify() # Pillowがサポートする形式か、破損していないか簡易チェック
                        img.close()
                    except Exception as img_err:
                        await processing_msg.edit(content=f"ファイル '{attachment.filename}' は有効な画像ファイルではないか、破損しているようです。処理を中断します。({img_err})")
                        os.remove(temp_save_path) # 一時ファイルを削除
                        continue
                # ここでmime_typeを正しく渡すことが重要
                mime_type_for_gemini = attachment.content_type # Discordが提供するMIMEタイプを使用
                tags_str = await get_tags_from_gemini(temp_save_path, attachment.filename, mime_type_for_gemini)
            except Exception as e:
                print(f"タグ付け処理中にエラー: {e}")
                await processing_msg.edit(content=f"ファイル '{attachment.filename}' のタグ付け中にエラーが発生しました。タグなしで保存します。")
                tags_str = "notags"


            # ファイル名命名規則 (要件定義書 3.2)
            date_str = datetime.datetime.now().strftime("%Y%m%d")
            original_filename_no_ext, original_ext = os.path.splitext(attachment.filename)
            sanitized_original_filename = sanitize_filename_component(original_filename_no_ext)

            new_filename = f"{date_str}_{tags_str}_{sanitized_original_filename}{original_ext}"
            final_save_path = os.path.join(year_month_folder_path, new_filename)

            # tempファイルをリネーム
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
                if os.path.exists(temp_save_path): # tempファイルが残っていれば削除
                    os.remove(temp_save_path)

    await bot.process_commands(message) # メッセージがコマンドであるかどうかもチェック

# --- スラッシュコマンド ---

# --- ファイル管理コマンド ---
@bot.tree.command(name="upload_guide", description="ファイルアップロード方法の案内")
async def upload_guide(interaction: discord.Interaction):
    await interaction.response.send_message(
        "ファイルをアップロードするには、このチャンネルに直接ファイルをドラッグ＆ドロップするか、メッセージ入力欄の「+」ボタンからファイルを添付して送信してください。\n"
        "画像または動画ファイルが対象です。",
        ephemeral=True # コマンド実行者のみに見えるメッセージ
    )

@bot.tree.command(name="list_files", description="保存されているファイルの一覧を表示します。年月やキーワードで絞り込み可能。")
@discord.app_commands.describe(year_month="表示したい年月 (例: 202505)。省略すると全期間。", keyword="ファイル名に含まれる検索キーワード。")
async def list_files(interaction: discord.Interaction, year_month: str = None, keyword: str = None):
    await interaction.response.defer(ephemeral=True) # 処理に時間がかかる可能性があるのでdefer

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
    else: # year_monthが指定されていない場合は全年月フォルダを対象
        for item in os.listdir(BASE_UPLOAD_FOLDER):
            item_path = os.path.join(BASE_UPLOAD_FOLDER, item)
            if os.path.isdir(item_path) and re.fullmatch(r"\d{6}", item): # YYYYMM形式のフォルダのみ
                search_folders.append(item_path)

    if not search_folders and not year_month: # フォルダが一つもなければ
        await interaction.followup.send("まだアップロードされたファイルはありません。")
        return
    elif not search_folders and year_month: # 指定年月フォルダがない場合 (上でもチェックしてるが一応)
         await interaction.followup.send(f"`{year_month}` に該当するファイルは見つかりませんでした。")
         return


    for folder in search_folders:
        for filename in os.listdir(folder):
            if os.path.isfile(os.path.join(folder, filename)):
                if keyword: # キーワード検索がある場合
                    if keyword.lower() in filename.lower():
                        found_files.append(filename)
                else: # キーワード検索がない場合
                    found_files.append(filename)

    if not found_files:
        msg = "該当するファイルは見つかりませんでした。"
        if keyword:
            msg += f" (キーワード: `{keyword}`)"
        if year_month:
            msg += f" (年月: `{year_month}`)"
        await interaction.followup.send(msg)
        return

    # メッセージチャンク制限を考慮して複数に分けて送信
    response_message = f"ファイル一覧 ({len(found_files)}件):\n"
    if keyword: response_message += f"検索キーワード: `{keyword}`\n"
    if year_month: response_message += f"年月: `{year_month}`\n"
    response_message += "```\n"

    current_length = len(response_message)
    files_in_chunk = 0
    for filename in sorted(found_files): # ソートして表示
        file_ext = os.path.splitext(filename)[1]
        icon = get_file_icon(file_ext)
        line = f"{icon} {filename}\n"
        if current_length + len(line) > 1980: # Discordのメッセージ長制限より少し手前で分割
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
    await list_files.callback(self=bot, interaction=interaction, year_month=None, keyword=keyword)


@bot.tree.command(name="download_file", description="指定されたファイル名のファイルをダウンロードします。")
@discord.app_commands.describe(filename="ダウンロードしたい正確なファイル名。")
async def download_file(interaction: discord.Interaction, filename: str):
    await interaction.response.defer(ephemeral=False) # ファイル送信は公開で良いだろう

    found_path = None
    # 全年月フォルダを検索
    for ym_folder_name in os.listdir(BASE_UPLOAD_FOLDER):
        ym_folder_path = os.path.join(BASE_UPLOAD_FOLDER, ym_folder_name)
        if os.path.isdir(ym_folder_path) and re.fullmatch(r"\d{6}", ym_folder_name):
            target_file_path = os.path.join(ym_folder_path, filename)
            if os.path.exists(target_file_path) and os.path.isfile(target_file_path):
                found_path = target_file_path
                break

    if found_path:
        try:
            # Discordのファイルサイズ制限に注意。BOTはNitro扱いではないため、通常8MB。
            # より大きなファイルは分割送信や外部ストレージリンクを検討する必要がある。
            file_size = os.path.getsize(found_path)
            if file_size > 8 * 1024 * 1024: # 8MB
                 await interaction.followup.send(
                    f"ファイル '{filename}' ({file_size // 1024 // 1024}MB) はサイズが大きすぎるため、直接送信できません。\n"
                    "（管理者の方へ: 将来的に外部ストレージ連携などの対応をご検討ください）"
                )
                 return

            await interaction.followup.send(f"ファイル '{filename}' を送信します:", file=discord.File(found_path))
        except Exception as e:
            print(f"ファイル送信エラー: {e}")
            await interaction.followup.send(f"ファイル '{filename}' の送信中にエラーが発生しました。")
    else:
        await interaction.followup.send(f"ファイル '{filename}' が見つかりませんでした。ファイル名は完全一致で入力してください。`/list_files` で確認できます。")


@bot.tree.command(name="edit_tags", description="ファイルのタグを編集します（ファイル名リネーム）。")
@discord.app_commands.describe(current_filename="現在の完全なファイル名。", new_tags="新しいタグ (カンマ区切り、例: タグ1,タグ2,タグ3)。タグなしは notags と入力。")
@is_admin() # 管理者のみ実行可能
async def edit_tags(interaction: discord.Interaction, current_filename: str, new_tags: str):
    await interaction.response.defer(ephemeral=True)

    current_filepath = None
    original_ym_folder_path = None

    # ファイルを探す
    for ym_folder_name in os.listdir(BASE_UPLOAD_FOLDER):
        ym_folder_path_loop = os.path.join(BASE_UPLOAD_FOLDER, ym_folder_name)
        if os.path.isdir(ym_folder_path_loop) and re.fullmatch(r"\d{6}", ym_folder_name):
            prospective_path = os.path.join(ym_folder_path_loop, current_filename)
            if os.path.exists(prospective_path) and os.path.isfile(prospective_path):
                current_filepath = prospective_path
                original_ym_folder_path = ym_folder_path_loop
                break

    if not current_filepath:
        await interaction.followup.send(f"ファイル '{current_filename}' が見つかりませんでした。")
        return

    # 新しいタグを処理
    if new_tags.strip().lower() == "notags":
        processed_new_tags = "notags"
    else:
        tags_list = [sanitize_filename_component(tag.strip()) for tag in new_tags.split(',') if tag.strip()]
        if not tags_list:
            await interaction.followup.send("新しいタグが指定されていません。タグなしにする場合は `notags` と入力してください。")
            return
        processed_new_tags = "-".join(tags_list)


    # 元のファイル名から日付と元のファイル名部分を抽出
    # 命名規則: [日付]_[タグセクション]_[元のファイル名].[拡張子]
    parts = current_filename.split('_', 2) # 最大2回分割
    if len(parts) < 3:
        await interaction.followup.send(f"ファイル '{current_filename}' は期待される命名規則に従っていません。手動でのリネームが必要かもしれません。")
        return

    date_str = parts[0]
    # parts[1] は古いタグセクション
    original_name_with_ext = parts[2]
    # original_filename_no_ext, original_ext = os.path.splitext(original_name_with_ext) # これだと元のファイル名にアンダースコアがあった場合破綻する

    # より堅牢な元のファイル名と拡張子の分離
    # 最後の'.'を基準に拡張子を分離し、それより前を「元のファイル名部分」とする
    # ただし、タグセクションの後に続くのが元のファイル名部分なので、parts[2]を使う
    base, ext = os.path.splitext(original_name_with_ext)
    # この base が [元のファイル名] に相当する。サニタイズ済みのはず。

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
@discord.app_commands.describe(filename="削除したい正確なファイル名。")
@is_admin() # 管理者のみ実行可能
async def delete_file(interaction: discord.Interaction, filename: str):
    await interaction.response.defer(ephemeral=True)

    found_path = None
    # 全年月フォルダを検索
    for ym_folder_name in os.listdir(BASE_UPLOAD_FOLDER):
        ym_folder_path = os.path.join(BASE_UPLOAD_FOLDER, ym_folder_name)
        if os.path.isdir(ym_folder_path) and re.fullmatch(r"\d{6}", ym_folder_name):
            target_file_path = os.path.join(ym_folder_path, filename)
            if os.path.exists(target_file_path) and os.path.isfile(target_file_path):
                found_path = target_file_path
                break

    if found_path:
        # 確認ステップの代わりに、ここではUI Select Viewを使った確認方法を実装
        view = ConfirmDeleteView(found_path, filename, interaction.user)
        await interaction.followup.send(
            f"本当にファイル `{filename}` を削除しますか？この操作は取り消せません。",
            view=view,
            ephemeral=True
        )
        await view.wait() # ユーザーの応答を待つ
        # 実際の削除はConfirmDeleteView内のボタンで行う
    else:
        await interaction.followup.send(f"ファイル '{filename}' が見つかりませんでした。ファイル名は完全一致で入力してください。")

# ファイル削除確認用のView
class ConfirmDeleteView(discord.ui.View):
    def __init__(self, filepath: str, filename: str, author: discord.User, timeout=30.0):
        super().__init__(timeout=timeout)
        self.filepath = filepath
        self.filename = filename
        self.author = author
        self.deleted = False

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        # このインタラクションが元のコマンド実行者からのものか確認
        if interaction.user.id != self.author.id:
            await interaction.response.send_message("この操作はコマンドを実行した本人のみが行えます。", ephemeral=True)
            return False
        return True

    @discord.ui.button(label="はい、削除します", style=discord.ButtonStyle.danger, custom_id="confirm_delete")
    async def confirm_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        try:
            os.remove(self.filepath)
            self.deleted = True
            print(f"ファイル '{self.filename}' をユーザー '{interaction.user}' の指示により削除しました。")
            await interaction.response.edit_message(content=f"ファイル `{self.filename}` を削除しました。", view=None)
        except Exception as e:
            print(f"ファイル削除エラー ({self.filename}): {e}")
            await interaction.response.edit_message(content=f"ファイル `{self.filename}` の削除中にエラーが発生しました。", view=None)
        self.stop() # Viewを停止

    @discord.ui.button(label="いいえ、キャンセル", style=discord.ButtonStyle.secondary, custom_id="cancel_delete")
    async def cancel_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.edit_message(content="ファイル削除をキャンセルしました。", view=None)
        self.stop()

    async def on_timeout(self):
        # タイムアウトした場合、メッセージを編集してボタンを無効化する
        # interaction.edit_original_response() などを使う必要があるが、
        # interactionオブジェクトがこのスコープにないので、元のメッセージを編集するのは少し工夫がいる。
        # ここでは何もしないか、元のinteractionを保持しておく必要がある。
        # フォローアップメッセージの編集は interaction.followup.edit_message(message_id=...)
        # 今回はシンプルに何もしないでおく (ボタンが押せなくなるだけ)
        print(f"ファイル '{self.filename}' の削除確認がタイムアウトしました。")
        # ボタンを無効化するためにメッセージを編集 (もしメッセージIDが分かれば)
        # view = self
        # for child in view.children: # type: ignore
        #     child.disabled = True
        # await interaction.edit_original_response(view=view) # ここで interaction がない
        pass


# --- BOT設定コマンド ---
@bot.tree.command(name="set_model", description="自動タグ付けに使用するGeminiモデルを設定します。")
@discord.app_commands.describe(model_name="Geminiモデル名 (例: gemini-1.5-flash-latest, gemini-1.5-pro-latest)。")
@is_admin()
async def set_model(interaction: discord.Interaction, model_name: str):
    global current_gemini_model, gemini_model_instance
    # 利用可能なモデルのリスト (実際のAPIで確認するのが望ましいが、ここでは代表的なものを例示) [cite: 3]
    # Gemini 1.5 Flash (gemini-1.5-flash-latest など)
    # Gemini 1.5 Pro (gemini-1.5-pro-latest など)
    # ドキュメントの命名規則参照 [cite: 4]
    # 例: gemini-2.5-pro-preview-05-06, gemini-2.5-flash-preview-05-20, gemini-2.0-flash
    # 簡単なバリデーション (実際にはAPIに問い合わせて存在確認するのがベスト)
    if not model_name.startswith("gemini-"):
        await interaction.response.send_message(
            f"モデル名が無効です。`gemini-` で始まるモデル名を入力してください (例: `gemini-1.5-flash-latest`)。",
            ephemeral=True
        )
        return

    try:
        # 新しいモデルでインスタンスを再作成
        new_model_instance = genai.GenerativeModel(
            model_name,
            safety_settings={ # 安全性設定も再度適用
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        # テストで簡単なリクエストを投げてみる (任意)
        # await new_model_instance.generate_content_async("test", generation_config={"response_mime_type": "text/plain"})

        current_gemini_model = model_name
        gemini_model_instance = new_model_instance # グローバル変数を更新
        await interaction.response.send_message(f"自動タグ付けに使用するGeminiモデルを `{model_name}` に設定しました。", ephemeral=True)
        print(f"Geminiモデルが '{model_name}' に変更されました。")
    except Exception as e:
        await interaction.response.send_message(f"モデル `{model_name}` の設定に失敗しました: {e}", ephemeral=True)
        print(f"Geminiモデル '{model_name}' の設定失敗: {e}")


@bot.tree.command(name="current_model", description="現在設定されているGeminiモデル名を表示します。")
async def current_model(interaction: discord.Interaction):
    await interaction.response.send_message(f"現在設定されているGeminiモデルは `{current_gemini_model}` です。", ephemeral=True)


@bot.tree.command(name="help_nasbot", description="このBOTのコマンド一覧と簡単な説明を表示します。")
async def help_nasbot(interaction: discord.Interaction):
    embed = discord.Embed(title="ファイル管理BOT ヘルプ", description="このBOTで利用可能なコマンド一覧です。", color=discord.Color.blue())
    embed.add_field(name="ファイル操作", value=(
        "`/upload_guide` - ファイルのアップロード方法を表示します。\n"
        "`/list_files [年月] [キーワード]` - ファイル一覧を表示します。\n"
        "`/search_files <キーワード>` - ファイルを検索します。\n"
        "`/download_file <ファイル名>` - ファイルをダウンロードします。"
    ), inline=False)
    embed.add_field(name="管理者向けコマンド", value=(
        "`/edit_tags <現在のファイル名> <新しいタグ>` - ファイルのタグを編集します。\n"
        "`/delete_file <ファイル名>` - ファイルを削除します。\n"
        "`/set_model <モデル名>` - 自動タグ付けに使用するGeminiモデルを設定します。"
    ), inline=False)
    embed.add_field(name="その他", value=(
        "`/current_model` - 現在のGeminiモデル名を表示します。\n"
        "`/help_nasbot` - このヘルプを表示します。"
    ), inline=False)
    embed.set_footer(text="ファイルを直接このチャンネルにアップロードすることでも処理が開始されます。")
    await interaction.response.send_message(embed=embed, ephemeral=True)


# --- BOT実行 ---
if __name__ == "__main__":
    if DISCORD_BOT_TOKEN and GEMINI_API_KEY:
        bot.run(DISCORD_BOT_TOKEN)
    else:
        print("エラー: DISCORD_BOT_TOKEN または GEMINI_API_KEY が .envファイルに設定されていません。")