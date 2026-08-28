# monitor.ps1 — 康复LoRA流水线 实时监控窗口
# 放在 pipeline.py 同目录下运行；自动定位日志与产物目录。
# 用法:  powershell -NoExit -ExecutionPolicy Bypass -File monitor.ps1
# 功能:
#   1) 实时滚动 pipeline 日志(报错红/警告黄/成功绿)
#   2) 每 10 秒打印进度概览(清洗块数/QA块数/最终产出/进程存活)
#   3) 若 pipeline.py 进程消失, 红字报警"可能已结束或崩溃"
param(
    [string]$Log = ""
)

$ErrorActionPreference = "SilentlyContinue"
# 脚本所在目录即 pipeline 目录(换电脑也适用)
$dir = $PSScriptRoot
if (-not $dir) { $dir = Split-Path -Parent $MyInvocation.MyCommand.Definition }
if (-not $dir) { $dir = (Get-Location).Path }

# ---- 定位日志文件: 优先 logs/pipeline.log, 否则最新 _run*.log ----
if (-not $Log) {
    $cands = @()
    $lp = Join-Path $dir "logs\pipeline.log"
    if (Test-Path $lp) { $cands += $lp }
    $cands += (Get-ChildItem -Path $dir -Filter "_run*.log" -ErrorAction SilentlyContinue |
               Select-Object -ExpandProperty FullName)
    if ($cands.Count -eq 0) {
        Write-Host "未找到日志 (logs/pipeline.log 或 _run*.log)。请先启动 pipeline.py。" -ForegroundColor Red
        Write-Host "监控目录: $dir" -ForegroundColor DarkGray
        exit 1
    }
    $Log = ($cands | ForEach-Object { Get-Item $_ } |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
}

Write-Host "══════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   康复LoRA流水线 实时监控" -ForegroundColor Cyan
Write-Host "   日志: $Log" -ForegroundColor DarkGray
Write-Host "   Ctrl+C 退出本窗口(不影响后台 pipeline)" -ForegroundColor DarkGray
Write-Host "══════════════════════════════════════════════════" -ForegroundColor Cyan

$lastCount = 0
$loop = 0

function Test-PipelineRunning {
    $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
             Where-Object { $_.CommandLine -like '*pipeline.py*' }
    return ($null -ne $procs)
}

function Show-Progress {
    $out = Join-Path $dir "output"
    $cleaned = 0; $qa = 0
    if (Test-Path (Join-Path $out "cleaned")) {
        $cleaned = (Get-ChildItem -Path (Join-Path $out "cleaned") -Recurse -Filter *.txt).Count
    }
    if (Test-Path (Join-Path $out "qa_raw")) {
        $qa = (Get-ChildItem -Path (Join-Path $out "qa_raw") -Recurse -Filter *.json).Count
    }
    $finalPath = Join-Path $out "final\rehab_lora_train_data.json"
    $final = Test-Path $finalPath
    $running = Test-PipelineRunning
    if ($running) {
        Write-Host "`n── 进度 [运行中] ──" -ForegroundColor Green
    } else {
        Write-Host "`n── 进度 [⚠️ 未检测到 pipeline.py 进程, 可能已结束或崩溃] ──" -ForegroundColor Red
    }
    $fin = if ($final) { "已生成 ✅" } else { "未生成" }
    Write-Host "   清洗块: $cleaned | QA块: $qa | 最终产出: $fin" -ForegroundColor White
}

while ($true) {
    if (Test-Path $Log) {
        $lines = Get-Content -Path $Log -Encoding utf8 -ReadCount 0
        if ($null -eq $lines) { $lines = @() }
        if ($lines -is [string]) { $lines = @($lines) }
        if ($lines.Count -gt $lastCount) {
            $new = $lines[$lastCount..($lines.Count - 1)]
            foreach ($l in $new) {
                $fg = "White"
                if ($l -match "❌|🔴|Error|Exception|Traceback|失败|timed out|超时|400|500|502|503") {
                    $fg = "Red"
                } elseif ($l -match "⚠️|WARNING|重试|retry|网络异常") {
                    $fg = "Yellow"
                } elseif ($l -match "✅|完成|跳过|🎉|✔|阶段") {
                    $fg = "Green"
                }
                Write-Host $l -ForegroundColor $fg
            }
            $lastCount = $lines.Count
        }
    }
    $loop++
    if ($loop % 10 -eq 0) { Show-Progress }
    Start-Sleep -Seconds 1
}
