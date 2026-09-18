import os
import sys
import shutil
import kagglehub

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def download_and_setup_data():
    """
    Kagglehub를 통해 heptapod/titanic 데이터셋을 다운로드하고
    프로젝트 내 data/ 폴더로 복사하여 정리합니다.
    """
    # 1. Kagglehub 캐시 디렉토리에 데이터셋 다운로드
    cache_path = kagglehub.dataset_download("heptapod/titanic")
    print(f"[+] Kagglehub 캐시 경로: {cache_path}")

    # 2. 프로젝트 루트 및 data 폴더 경로 설정
    project_root = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(project_root, "data")
    os.makedirs(target_dir, exist_ok=True)

    # 3. 캐시 폴더의 파일들을 프로젝트 data/ 폴더로 복사
    copied_files = []
    for item in os.listdir(cache_path):
        src_path = os.path.join(cache_path, item)
        dst_path = os.path.join(target_dir, item)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
            copied_files.append(item)
            print(f"[+] 파일 복사 완료: {item} -> {dst_path}")
        elif os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            copied_files.append(item)
            print(f"[+] 디렉토리 복사 완료: {item} -> {dst_path}")

    print("\n[V] 데이터셋이 프로젝트 data/ 폴더에 성공적으로 정리되었습니다.")
    return target_dir, copied_files

if __name__ == "__main__":
    download_and_setup_data()
