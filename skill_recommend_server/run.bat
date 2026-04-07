@echo off
setlocal

cd /d "%~dp0"

python -m skills_recommender %* --api --update-skills

endlocal
