# ex. python audio_to_wav.py -s ../raw_data/수사기관사칭형_VIDEO/ -d ../raw_data/수사기관사칭형_WAV -e 수사기관사칭형_to_wav_errors.csv
import os
import subprocess
import argparse
import functools
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import time
import soundfile as sf

MEDIA_EXTENSIONS = ('.m4a', '.mp3', '.wav', '.amr', '.3gp', '.3gpp', '.mp4', '.aac')

def format_duration(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}시간 {minutes:02d}분 {secs:02d}초"

def calculate_total_duration(wav_root_dir):

    print("\n변환된 .wav 파일의 총 오디오 길이")
    wav_files = list(Path(wav_root_dir).rglob('*.wav'))

    total_duration_sec = 0
    with tqdm(total=len(wav_files), desc="Calculating duration") as pbar:
        for wav_path in wav_files:
            try:
                info = sf.info(str(wav_path))
                total_duration_sec += info.duration
            except Exception as e:
                print(f"'{wav_path.name}' 파일 길이 읽기 실패: {e}")
            pbar.update(1)
            
    return total_duration_sec

def find_all_media_files(root_dir):
    """지정된 디렉토리에서 모든 미디어 파일을 재귀적으로 탐색"""
    media_paths = []
    print(f"'{root_dir}'의 미디어 파일")
    for extension in MEDIA_EXTENSIONS:
        media_paths.extend(Path(root_dir).rglob(f'*{extension}'))
    print(f"총 {len(media_paths)}개")
    return media_paths

def convert_to_wav(source_path: Path, source_root: Path, dest_root: Path):
    """
    단일 미디어 파일을 .wav로 변환
    """
    try:
        relative_path = source_path.relative_to(source_root)
        dest_path = dest_root / relative_path.with_suffix('.wav')
        
        # if dest_path.exists():
        #     return ('SKIPPED', str(source_path))

        dest_path.parent.mkdir(parents=True, exist_ok=True)

        probe_command = ['ffprobe', '-v', 'error', str(source_path)]
        subprocess.run(probe_command, check=True, capture_output=True, text=True)

        convert_command = [
            'ffmpeg',
            '-i', str(source_path),
            '-vn', '-ar', '16000', '-ac', '1',
            '-c:a', 'pcm_s16le',
            '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11',
            str(dest_path),
            '-y', '-loglevel', 'error'
        ]
        subprocess.run(convert_command, check=True, capture_output=True, text=True)
        return ('SUCCESS', str(source_path))

    except subprocess.CalledProcessError as e:
        error_details = e.stderr.strip() if e.stderr else e.stdout.strip()
        error_info = {
            'filepath': str(source_path),
            'error_type': 'Process Error (ffmpeg/ffprobe)',
            'error_message': 'A command-line process failed.',
            'details': error_details
        }
        return ('ERROR', error_info)
    except Exception as e:
        error_info = {
            'filepath': str(source_path),
            'error_type': 'Unknown Error',
            'error_message': 'An unexpected Python error occurred.',
            'details': str(e)
        }
        return ('ERROR', error_info)

def main(args):
    
    start_time = time.time()
    
    source_root_path = Path(args.source_dir)
    dest_root_path = Path(args.dest_dir)
    error_csv_path = args.error_csv

    if not source_root_path.is_dir():
        print(f"오류: 소스 디렉토리X - {source_root_path}")
        return

    source_files = find_all_media_files(source_root_path)

    if not source_files:
        print("처리할 파일 X")
    else:
        num_processes = cpu_count()
        print(f"{num_processes}개의 프로세스로 변환 시작")
        
        errors = []
        success_count = 0
        skipped_count = 0 

        convert_task = functools.partial(
            convert_to_wav, 
            source_root=source_root_path, 
            dest_root=dest_root_path
        )

        with Pool(processes=num_processes) as pool:
            with tqdm(total=len(source_files), desc="Converting files") as pbar:
                for status, info in pool.imap_unordered(convert_task, source_files):
                    if status == 'SUCCESS':
                        success_count += 1
                    elif status == 'SKIPPED':
                        skipped_count += 1
                    elif status == 'ERROR':
                        errors.append(info)
                    pbar.update(1)

        print("\n변환 작업 요약")
        print(f"총 대상 파일: {len(source_files)} 개")
        print(f"새로 변환 성공: {success_count} 개")
        print(f"건너뜀 (이미 변환됨): {skipped_count} 개")
        print(f"실패: {len(errors)} 개")
        
        if errors:
            print("\n변환 중 오류가 발생한 파일 상세 정보")
            error_df = pd.DataFrame(errors)
            for index, row in error_df.iterrows():
                print(f"\n[{index + 1}] 파일: {row['filepath']}")
                print(f"   - 오류 타입: {row['error_type']}")
                print(f"   - 상세 내용: {row['details']}")

            error_df.to_csv(error_csv_path, index=False, encoding='utf-8-sig')
            print(f"\n\n  오류 발생 파일 상세 내역 '{error_csv_path}' 파일에 저장")
        
        print(f"\n오디오/비디오 파일 변환 완료")
        print(f"변환된 파일 '{dest_root_path}' 저장")

    
    # 변환 작업이 모두 끝난 후, 대상 디렉토리의 모든 .wav 파일 길이를 계산
    try:
        total_length_sec = calculate_total_duration(dest_root_path)
        print("\n총 오디오 길이 (전체 .wav)")
        print(f"  {format_duration(total_length_sec)}")
    except Exception as e:
        print(f"\n총 오디오 길이 계산 중 오류 발생: {e}")


    end_time = time.time()
    print(f"\n총 소요 시간")
    print(f"  {format_duration(end_time - start_time)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""미디어 파일을 16kHz 모노 .wav 파일로 변환"""
    )
    
    parser.add_argument(
        '-s', '--source_dir', 
        required=True, 
        type=str,
        help="변환할 원본 미디어 파일이 있는 루트 디렉토리"
    )
    
    parser.add_argument(
        '-d', '--dest_dir', 
        required=True, 
        type=str,
        help="변환된 .wav 파일을 저장할 루트 디렉토리 (원본 구조 유지)"
    )
    
    parser.add_argument(
        '-e', '--error_csv', 
        default='audio_to_wav_errors.csv', 
        type=str,
        help="변환 오류 로그를 저장할 CSV 파일 경로"
    )

    try:
        args = parser.parse_args()
        main(args)
    except ImportError:
        # main 함수 이전에 soundfile 임포트 실패 시 여기서 처리
        pass 
    except Exception as e:
        print(f"\n스크립트 실행 중 오류 발생: {e}")