@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM MiniMind book pipeline (6GB GPU profile)
REM Steps:
REM   1) Build datasets from PDF/EPUB
REM   2) Pretrain
REM   3) Full SFT
REM   4) Inference smoke test
REM   5) Fixed QA evaluation (if eval file exists)
REM ============================================================

chcp 65001 >nul

REM -------- User paths --------
set "PROJECT_DIR=C:\Users\dereckyin\Desktop\minimind"
set "BOOKS_DIR=Y:\backup"

REM -------- Run tag --------
set "RUN_TAG=books_6g_v1"
set "PRETRAIN_WEIGHT=pretrain_%RUN_TAG%"
set "SFT_WEIGHT=full_sft_%RUN_TAG%"

REM -------- Dataset outputs --------
set "PRETRAIN_JSON=dataset\books_pretrain_%RUN_TAG%.jsonl"
set "SFT_JSON=dataset\books_sft_seed_%RUN_TAG%.jsonl"
set "SFT_CLEAN_JSON=dataset\books_sft_clean_%RUN_TAG%.jsonl"
set "SFT_CLEAN_REPORT=dataset\books_sft_clean_report_%RUN_TAG%.json"
set "REPORT_JSON=dataset\books_build_report_%RUN_TAG%.json"

REM -------- Build dataset params --------
set "MAX_CHARS=1200"
set "MIN_CHARS=200"
set "PAIRS_PER_BOOK=24"
set "MAX_ANSWER_CHARS=320"
set "MAX_BOOKS=5000"
set "EARLY_STOP_SCAN=1"
set "PDF_MAX_PAGES=50"
set "PDF_MAX_SECONDS=12"
set "BUILD_START_INDEX=0"
set "BUILD_END_INDEX=-1"
set "BUILD_APPEND=0"
set "REPORT_PER_BOOK=0"
set "LOG_INTERVAL=100"
set "SEARCH_DEPTH=1"

REM -------- 6GB training profile --------
set "HIDDEN_SIZE=192"
set "NUM_LAYERS=6"
set "EPOCHS_PRETRAIN=1"
set "EPOCHS_SFT=1"
set "BATCH_SIZE=1"
set "ACC_STEPS=16"
set "MAX_SEQ_LEN_PRE=128"
set "MAX_SEQ_LEN_SFT=160"
set "LR_PRETRAIN=4e-4"
set "LR_SFT=2e-5"
set "NUM_WORKERS=2"

REM -------- Quick test --------
set "USE_EXISTING_DATASET=0"
set "ENABLE_SFT_CLEAN=1"

REM -------- Runtime env --------
set "TRANSFORMERS_NO_TF=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"

echo.
echo [INFO] PROJECT_DIR = %PROJECT_DIR%
echo [INFO] BOOKS_DIR   = %BOOKS_DIR%
echo.

if not exist "%PROJECT_DIR%" (
  echo [ERROR] PROJECT_DIR not found: %PROJECT_DIR%
  exit /b 1
)
if not exist "%BOOKS_DIR%" (
  echo [ERROR] BOOKS_DIR not found: %BOOKS_DIR%
  exit /b 1
)

pushd "%PROJECT_DIR%"

echo [STEP 0/5] Environment check...
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
if errorlevel 1 (
  echo [ERROR] Python/Torch check failed.
  popd
  exit /b 1
)

python -c "import importlib.util; print('pymupdf', 'ok' if importlib.util.find_spec('fitz') else 'missing')"
if errorlevel 1 (
  echo [WARN] Could not verify PyMuPDF.
)

for /f "delims=" %%i in ('python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"') do set "DEVICE=%%i"
if not defined DEVICE set "DEVICE=cpu"
if "%DEVICE%"=="cpu" (set "DTYPE=float32") else (set "DTYPE=float16")
echo [INFO] Using device: %DEVICE%, dtype: %DTYPE%

echo.
echo [STEP 1/5] Build datasets from books...
if "%USE_EXISTING_DATASET%"=="1" (
  echo [INFO] USE_EXISTING_DATASET=1, using dataset\pretrain_hq.jsonl and dataset\sft_mini_512.jsonl
  set "PRETRAIN_JSON=dataset\pretrain_hq.jsonl"
  set "SFT_JSON=dataset\sft_mini_512.jsonl"
) else (
  set "BUILD_APPEND_FLAG="
  if "%BUILD_APPEND%"=="1" (
    set "BUILD_APPEND_FLAG=--append"
    echo [INFO] BUILD_APPEND=1, resuming from start_index=%BUILD_START_INDEX%
  )
  python scripts\build_books_dataset.py ^
    --input_dir "%BOOKS_DIR%" ^
    --pretrain_out "%PRETRAIN_JSON%" ^
    --sft_out "%SFT_JSON%" ^
    --report_out "%REPORT_JSON%" ^
    --max_books %MAX_BOOKS% ^
    --max_chars %MAX_CHARS% ^
    --min_chars %MIN_CHARS% ^
    --pairs_per_book %PAIRS_PER_BOOK% ^
    --max_answer_chars %MAX_ANSWER_CHARS% ^
    --start_index %BUILD_START_INDEX% ^
    --end_index %BUILD_END_INDEX% ^
    --report_per_book %REPORT_PER_BOOK% ^
    --log_interval %LOG_INTERVAL% ^
    --search_depth %SEARCH_DEPTH% ^
    --pdf_max_pages %PDF_MAX_PAGES% ^
    --pdf_max_seconds %PDF_MAX_SECONDS% ^
    --early_stop_scan %EARLY_STOP_SCAN% !BUILD_APPEND_FLAG!
  if errorlevel 1 (
    echo [ERROR] Dataset build failed.
    popd
    exit /b 1
  )
  if "%ENABLE_SFT_CLEAN%"=="1" (
    echo.
    echo [STEP 1.5/5] Clean SFT seed dataset...
    python scripts\clean_books_sft.py ^
      --input "%SFT_JSON%" ^
      --output "%SFT_CLEAN_JSON%" ^
      --report "%SFT_CLEAN_REPORT%"
    if errorlevel 1 (
      echo [ERROR] SFT clean step failed.
      popd
      exit /b 1
    )
    set "SFT_JSON=%SFT_CLEAN_JSON%"
  )
)

echo.
echo [STEP 2/5] Pretrain...
pushd trainer
python train_pretrain.py ^
  --epochs %EPOCHS_PRETRAIN% ^
  --batch_size %BATCH_SIZE% ^
  --learning_rate %LR_PRETRAIN% ^
  --device %DEVICE% ^
  --dtype %DTYPE% ^
  --num_workers %NUM_WORKERS% ^
  --accumulation_steps %ACC_STEPS% ^
  --log_interval 20 ^
  --save_interval 200 ^
  --hidden_size %HIDDEN_SIZE% ^
  --num_hidden_layers %NUM_LAYERS% ^
  --max_seq_len %MAX_SEQ_LEN_PRE% ^
  --data_path ..\%PRETRAIN_JSON% ^
  --save_weight %PRETRAIN_WEIGHT% ^
  --from_weight none
if errorlevel 1 (
  echo [ERROR] Pretrain failed.
  popd
  popd
  exit /b 1
)

echo.
echo [STEP 3/5] Full SFT...
python train_full_sft.py ^
  --epochs %EPOCHS_SFT% ^
  --batch_size %BATCH_SIZE% ^
  --learning_rate %LR_SFT% ^
  --device %DEVICE% ^
  --dtype %DTYPE% ^
  --num_workers %NUM_WORKERS% ^
  --accumulation_steps %ACC_STEPS% ^
  --log_interval 20 ^
  --save_interval 200 ^
  --hidden_size %HIDDEN_SIZE% ^
  --num_hidden_layers %NUM_LAYERS% ^
  --max_seq_len %MAX_SEQ_LEN_SFT% ^
  --data_path ..\%SFT_JSON% ^
  --from_weight %PRETRAIN_WEIGHT% ^
  --save_weight %SFT_WEIGHT%
if errorlevel 1 (
  echo [ERROR] Full SFT failed.
  popd
  popd
  exit /b 1
)
popd

echo.
echo [STEP 4/5] Inference smoke test...
(echo 0) | python eval_llm.py ^
  --load_from model ^
  --save_dir out ^
  --weight %SFT_WEIGHT% ^
  --hidden_size %HIDDEN_SIZE% ^
  --num_hidden_layers %NUM_LAYERS% ^
  --device %DEVICE% ^
  --max_new_tokens 64 ^
  --show_speed 0
if errorlevel 1 (
  echo [WARN] Inference smoke test failed.
)

echo.
echo [STEP 5/5] Fixed QA evaluation...
if exist "dataset\eval_fixed_qa.jsonl" (
  python scripts\eval_fixed_qa.py ^
    --tokenizer_path model ^
    --weight_path "out\%SFT_WEIGHT%_%HIDDEN_SIZE%.pth" ^
    --hidden_size %HIDDEN_SIZE% ^
    --num_hidden_layers %NUM_LAYERS% ^
    --eval_data dataset\eval_fixed_qa.jsonl ^
    --device %DEVICE% ^
    --save_report "out\eval_fixed_qa_report_%RUN_TAG%.json"
  if errorlevel 1 (
    echo [WARN] Fixed QA evaluation failed.
  )
) else (
  echo [INFO] Skip fixed QA eval: dataset\eval_fixed_qa.jsonl not found.
)

echo.
echo [DONE] Pipeline finished.
echo [INFO] Weights:
echo   out\%PRETRAIN_WEIGHT%_%HIDDEN_SIZE%.pth
echo   out\%SFT_WEIGHT%_%HIDDEN_SIZE%.pth
echo [INFO] Datasets:
echo   %PRETRAIN_JSON%
echo   %SFT_JSON%
echo [INFO] Build report:
echo   %REPORT_JSON%

popd
exit /b 0
