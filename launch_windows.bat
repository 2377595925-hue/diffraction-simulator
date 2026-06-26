@echo off
REM ============================================
REM  光学衍射仿真 — Windows 启动脚本
REM  双击此文件即可运行
REM ============================================

echo.
echo ============================================
echo   光学衍射仿真 — 3D 交互式界面
echo ============================================
echo.

REM 查找 Python
set PYTHON_BIN=
where python >nul 2>&1 && set PYTHON_BIN=python
where python3 >nul 2>&1 && set PYTHON_BIN=python3

if "%PYTHON_BIN%"=="" (
    echo ❌ 未找到 Python，请先安装 Python 3.8 或更高版本。
    echo    下载地址: https://www.python.org/downloads/windows/
    pause
    exit /b 1
)

echo 使用 Python: 
%PYTHON_BIN% --version
echo.

REM 检查依赖
echo 正在检查依赖...
%PYTHON_BIN% -c "import flask, numpy, scipy" 2>nul
if errorlevel 1 (
    echo ⚠️  缺少依赖，正在自动安装...
    %PYTHON_BIN% -m pip install flask numpy scipy pillow
    if errorlevel 1 (
        echo ❌ 依赖安装失败，请手动运行:
        echo   %PYTHON_BIN% -m pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo ✓ 依赖安装完成
) else (
    echo ✓ 依赖检查通过
)

echo.
echo 正在启动服务器...
echo （服务器启动后会自动打开浏览器）
echo.

REM 启动 diffraction_app.py
%PYTHON_BIN% diffraction_app.py

if errorlevel 1 (
    echo.
    echo 程序异常退出，请检查错误信息。
    pause
)
