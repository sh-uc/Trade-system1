from pathlib import Path

def convert_file_to_lf_pathlib(file_path):
    file = Path(file_path)
    content = file.read_text(encoding='utf-8').replace('\r\n', '\n').replace('\r', '\n')
    file.write_text(content, encoding='utf-8')
    print(f"Converted using pathlib: {file_path}")

# 対象ファイルを指定
target_file = "root/*.py"
convert_file_to_lf_pathlib(target_file)
