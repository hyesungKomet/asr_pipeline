"""
실행 예시
nohup python run_asr.py \
    --speech_data_root /home/hyesungkomet/VP/asr/raw_data/crawling_wav \
    --asr_result_root crawling_asr_result_dataset_chunked2 \
    --gpus 0 1 \
    --batch_size 4 \
    --log_file large-v3_eval_dataset_crawling_chunked2.log \
    --error_csv_path large-v3_error_log_dataset_crawling_chunked2.csv \
    --eval \
    --golden_json_path ../json_data/golden_answer.json \
    --output_csv_path large-v3_eval_dataset_crawling_chunked2.csv \
    > run_asr_chunked2.log 2>&1 &
"""

import os
import json
import pandas as pd
from tqdm import tqdm
import jiwer
import torch
from transformers import pipeline
from multiprocessing import Pool, Manager
import logging
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
import argparse

class AudioDataset(Dataset):
    """오디오 파일 경로 리스트 - PyTorch Dataset 클래스."""
    def __init__(self, audio_files):
        self.audio_files = audio_files

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        return self.audio_files[idx]

def setup_logging(logfile):
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logfile, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def find_files_to_process(speech_root, result_root):
    """
    처리할 오디오 파일 목록(아직 결과 파일이 없거나, 0.1초 이상인 파일)
    """
    files_to_process = []
    extensions = ('.wav',)
    
    all_audio_files = []
    for root, _, files in os.walk(speech_root):
        for file in files:
            if file.lower().endswith(extensions):
                all_audio_files.append(os.path.join(root, file))

    logging.info(f"총 {len(all_audio_files)}개 wav 파일 발견")

    for audio_path in tqdm(all_audio_files, desc="기존 결과 확인 & 오디오 길이 측정 중"):
        relative_path = os.path.relpath(audio_path, speech_root)
        expected_result_path = os.path.join(result_root, os.path.splitext(relative_path)[0] + '.txt')
        
        if not os.path.exists(expected_result_path):
            try:
                info = sf.info(audio_path)
                if info.duration > 0.1:
                    # (파일경로, 오디오 길이) 튜플
                    files_to_process.append((audio_path, info.duration))
                else:
                    logging.warning(f"0.1초 미만 오디오 파일 스킵: {audio_path}")
            except Exception as e:
                logging.error(f"오디오 파일 읽기 실패로 스킵: {audio_path} - 원인: {e}")

    return files_to_process

def load_existing_results(result_root):
    """기존 ASR(.txt) 결과들 로드"""
    existing_results = {}
    logging.info(f"'{result_root}'에서 기존 결과 로드 시작")
    
    txt_files = []
    for root, _, files in os.walk(result_root):
        for file in files:
            if file.lower().endswith('.txt'):
                txt_files.append(os.path.join(root, file))
                
    for txt_path in tqdm(txt_files, desc="기존 결과 로드 중"):
        file_id = os.path.splitext(os.path.basename(txt_path))[0]
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                existing_results[file_id] = f.read().strip()
        except Exception as e:
            logging.error(f"기존 결과 파일 로드 실패: {txt_path} - 원인: {e}")
            
    return existing_results

def run_asr_on_gpu(args_tuple):
    """단일 GPU ASR 처리"""
    try:
        (
            file_list, 
            gpu_id, 
            model_id, 
            batch_size, 
            num_workers, 
            speech_data_root, 
            asr_result_root,
            language,
            shared_error_list
        ) = args_tuple
        
        device = f"cuda:{gpu_id}"
        dtype = torch.float16
        local_results = {}
        
        pipe = pipeline(
            "automatic-speech-recognition", model=model_id, dtype=dtype, device=device
        )
    except Exception as e:
        logging.error(f"모델 로딩 실패 (GPU {gpu_id}): {e}")
        for audio_path in file_list:
            error_info = {
                'filepath': audio_path, 
                'filename': os.path.basename(audio_path), 
                'directory': os.path.dirname(audio_path), 
                'error_message': f"Model loading failed on GPU {gpu_id}: {e}"
            }
            shared_error_list.append(error_info)
        return None

    dataset = AudioDataset(file_list)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, # 정렬 순서를 유지해야 하므로 False
        num_workers=num_workers
    )

    for batch_files in tqdm(dataloader, desc=f"GPU {gpu_id} 처리 중", leave=False):
        try:
            outputs = pipe(
                batch_files, 
                return_timestamps=True, 
                generate_kwargs={"language": language},
                chunk_length_s=14,
                stride_length_s=(6,2),
            )

            for audio_path, result in zip(batch_files, outputs):
                file_id = os.path.splitext(os.path.basename(audio_path))[0]
                asr_text = result['text'].strip()
                local_results[file_id] = asr_text

                relative_path = os.path.relpath(audio_path, speech_data_root)
                result_path = os.path.join(asr_result_root, os.path.splitext(relative_path)[0] + '.txt')
                os.makedirs(os.path.dirname(result_path), exist_ok=True)
                with open(result_path, 'w', encoding='utf-8') as f:
                    f.write(asr_text)
        except Exception as e:
            logging.error(f"배치 처리 중 오류 발생 (GPU {gpu_id}): {batch_files} - 원인: {e}")
            for audio_path in batch_files:
                file_id = os.path.splitext(os.path.basename(audio_path))[0]
                local_results[file_id] = f"ERROR: {e}"
                error_info = {
                    'filepath': audio_path, 
                    'filename': os.path.basename(audio_path), 
                    'directory': os.path.dirname(audio_path), 
                    'error_message': str(e)
                }
                shared_error_list.append(error_info)
            continue
            
    return local_results

def process_files(args, files_to_run):
    """멀티프로세싱 -> 여러 GPU에 ASR 작업 분배"""
    manager = Manager()
    shared_error_list = manager.list()
    
    num_gpus = len(args.gpus)
    files_per_gpu = [files_to_run[i::num_gpus] for i in range(num_gpus)]
    
    # 각 워커에 전달할 인수 튜플 생성
    args_for_pool = [
        (
            files_per_gpu[i], 
            args.gpus[i], 
            args.model_id, 
            args.batch_size, 
            args.num_workers,
            args.speech_data_root,
            args.asr_result_root,
            args.language,
            shared_error_list
        ) for i in range(num_gpus)
    ]
    
    logging.info(f"{num_gpus}개의 GPU(ID: {args.gpus})로 ASR 작업 시작 (배치 크기: {args.batch_size}, 워커 수: {args.num_workers})")
    
    with Pool(processes=num_gpus) as pool:
        pool.map(run_asr_on_gpu, args_for_pool)
        
    logging.info("새로운 ASR 작업 완료")
    
    if shared_error_list:
        error_list = list(shared_error_list)
        logging.info(f"총 {len(error_list)}개 오류 발생. '{args.error_csv_path}' 파일에 저장")
        error_df = pd.DataFrame(error_list)
        error_df.to_csv(args.error_csv_path, index=False, encoding='utf-8-sig')

def evaluate_results(golden_json_path, all_asr_results, output_csv_path):
    """ASR 결과를 정답과 비교하여 WER/CER을 계산하고 CSV로 저장"""
    logging.info(f"정답 파일 '{golden_json_path}' 로드 시작.")
    try:
        with open(golden_json_path, 'r', encoding='utf-8') as f:
            golden_data = json.load(f)
        golden_dict = {item['id']: item['text'] for item in golden_data}
        logging.info(f"정답 파일 로드 완료. {len(golden_dict)}개 데이터 확인.")
    except Exception as e:
        logging.error(f"정답 파일 로드 실패: {e}")
        return

    evaluation_results = []
    
    for file_id, golden_text in tqdm(golden_dict.items(), desc="평가 진행 중"):
        asr_text = all_asr_results.get(file_id, "")
        
        # ASR 결과가 없거나, 정답 텍스트가 없거나, 에러가 발생한 경우 건너뜀
        if not asr_text or not golden_text or asr_text.startswith("ERROR:"):
            continue
            
        try:
            word_output = jiwer.process_words(golden_text, asr_text)
            char_error_rate = jiwer.cer(golden_text, asr_text)
            
            evaluation_results.append({
                'id': file_id, 'golden_text': golden_text, 'asr_transcript': asr_text,
                'wer': word_output.wer, 'wer_S': word_output.substitutions,
                'wer_D': word_output.deletions, 'wer_I': word_output.insertions,
                'cer': char_error_rate,
            })
        except Exception as e:
            logging.warning(f"ID {file_id} 평가 중 오류 발생: {e}")
    
    if evaluation_results:
        df = pd.DataFrame(evaluation_results)
        df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        logging.info(f"'{output_csv_path}' 파일에 평가 결과 저장 완료.")
    else:
        logging.warning("평가할 유효한 데이터가 없습니다.")

def parse_arguments():
    """CMD ARG 파싱"""
    parser = argparse.ArgumentParser(description="Whisper ASR 일괄 처리 및 평가 스크립트")
    
    # --- 필수 경로 ---
    parser.add_argument("--speech_data_root", type=str, required=True, 
                        help="원본 .wav 파일이 있는 루트 디렉토리")
    parser.add_argument("--asr_result_root", type=str, required=True, 
                        help="ASR 결과 .txt 파일을 저장할 디렉토리")
    
    # 모델 및 처리 설정
    parser.add_argument("--model_id", type=str, default="openai/whisper-large-v3", 
                        help="Hugging Face Whisper 모델 ID")
    parser.add_argument("--gpus", type=int, nargs='+', default=[0], 
                        help="사용할 GPU ID 리스트 (예: 0 1)")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="GPU당 배치 크기")
    parser.add_argument("--num_workers", type=int, default=0, 
                        help="DataLoader에서 사용할 워커 수")
    parser.add_argument("--language", type=str, default="korean", 
                        help="ASR 수행 언어")
    
    # 로깅
    parser.add_argument("--log_file", type=str, default="asr_processing.log", 
                        help="로그 파일 저장 경로")
    parser.add_argument("--error_csv_path", type=str, default="asr_error_log.csv", 
                        help="오류 로그 CSV 파일 저장 경로")
    
    # 평가(옵션)
    parser.add_argument("--eval", action="store_true", 
                        help="평가(WER/CER) 수행 여부")
    parser.add_argument("--golden_json_path", type=str, default="../json_data/golden_answer.json", 
                        help="평가를 위한 정답 JSON 파일 경로")
    parser.add_argument("--output_csv_path", type=str, default="asr_evaluation.csv", 
                        help="평가 결과 CSV 파일 저장 경로")
    
    args = parser.parse_args()
    
    if args.eval and (not args.golden_json_path or not args.output_csv_path):
        parser.error("--eval 사용 시 --golden_json_path와 --output_csv_path 모두 필요")
        
    return args

def main(args):
    """메인 함수."""
    setup_logging(args.log_file)
    logging.info("스크립트 시작")
    logging.info(f"실행 인수: {vars(args)}")

    # 1. 처리할 파일 찾기
    files_to_run_with_duration = find_files_to_process(args.speech_data_root, args.asr_result_root)
    
    if not files_to_run_with_duration:
        logging.info("새롭게 처리할 오디오 파일X")
    else:
        # 2. 파일 처리
        logging.info(f"총 {len(files_to_run_with_duration)}개 신규 파일 처리. 길이를 기준으로 정렬")
        # 오디오 길이를 기준으로 오름차순 정렬
        files_to_run_with_duration.sort(key=lambda x: x[1])
        # 정렬 후 파일 경로만 리스트로
        files_to_run = [path for path, duration in files_to_run_with_duration]
        
        process_files(args, files_to_run)
    
    # 3. 평가(--eval)
    # ASR 끝난 후 모든 결과를 로드
    all_asr_results = load_existing_results(args.asr_result_root)
    logging.info(f"총 {len(all_asr_results)}개의 ASR 결과 로드 완료.")
    
    if args.eval:
        logging.info("성능 평가 및 CSV 파일 생성 시작")
        evaluate_results(args.golden_json_path, all_asr_results, args.output_csv_path)
    else:
        logging.info("평가(--eval)가 설정되지 않아 평가 스킵")
    
    logging.info("스크립트 종료")

if __name__ == '__main__':
    # CUDA와 multiprocessing 사용 시 'spawn' 시작
    torch.multiprocessing.set_start_method('spawn', force=True)
    
    args = parse_arguments()
    main(args)