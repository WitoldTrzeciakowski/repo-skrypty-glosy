@echo off
echo Running script 1...
python resample_audio_and_clear_of_noise.py

if %ERRORLEVEL% NEQ 0 (
    echo Script 1 failed.
    exit /b 1
) else (
    echo Script 1 completed successfully.
)

echo Running script 2...
python silence_removal.py

if %ERRORLEVEL% NEQ 0 (
    echo Script 2 failed.
    exit /b 1
) else (
    echo Script 2 completed successfully.
)

echo Both scripts have been executed.
