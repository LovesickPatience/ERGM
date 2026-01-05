@echo off
REM Batch-extract 16 kHz mono WAV audio from all videos in a folder (Windows CMD).
REM Usage: extract_audio_windows.bat <video_dir> [out_dir]
REM Example: extract_audio_windows.bat "D:\MELD.Raw\train_splits" "data\audio\train"

if "%~1"=="" (
    echo Usage: %~nx0 ^<video_dir^> [out_dir]
    exit /b 1
)

set "VID_DIR=%~1"
set "OUT_DIR=%~2"
if "%OUT_DIR%"=="" set "OUT_DIR=audio"

if not exist "%OUT_DIR%" (
    mkdir "%OUT_DIR%"
)

REM Loop over mp4/mkv/avi (one level, adjust `for /r` to recurse)
for %%f in ("%VID_DIR%\*.mp4" "%VID_DIR%\*.mkv" "%VID_DIR%\*.avi") do (
    if exist "%%~f" (
        set "FILE=%%~nf"
        echo Extracting audio from: %%~f
        ffmpeg -y -i "%%~f" -vn -ar 16000 -ac 1 -c:a pcm_s16le "%OUT_DIR%\%%~nxf.wav"
    )
)

echo Done. WAV files saved to: %OUT_DIR%
