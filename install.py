import subprocess
import sys
import os

def install_requirements():
    """
    requirements.txt に基づいてライブラリをインストールする。
    """
    requirements_file = "requirements.txt"

    # requirements.txtが同じディレクトリに存在するか確認
    if not os.path.exists(requirements_file):
        print(f"エラー: '{requirements_file}' が見つかりません。")
        print("BOTと同じディレクトリに、必要なライブラリがリストされた requirements.txt を作成してください。")
        return

    try:
        print(f"'{requirements_file}' に基づいてライブラリをインストールします...")
        # pipを使ってrequirements.txtからインストール
        # sys.executable は現在実行中のPythonインタプリタのパスを指す
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        print("\nライブラリのインストール (または確認) が完了しました。")
        print("次は .env ファイルの設定を行い、python bot.py でBOTを起動してください。")

    except subprocess.CalledProcessError as e:
        print(f"エラー: ライブラリのインストール中に問題が発生しました。")
        print(f"コマンド: {e.cmd}")
        print(f"リターンコード: {e.returncode}")
        print(f"出力: {e.output}")
        print("\n手動で 'pip install -r requirements.txt' を試してみてください。")
    except FileNotFoundError:
        print("エラー: 'pip' コマンドが見つかりません。Pythonのインストール時に 'Add Python to PATH' にチェックを入れましたか？")
        print("または、Pythonの仮想環境を使用している場合は、仮想環境が有効になっているか確認してください。")

if __name__ == "__main__":
    install_requirements()