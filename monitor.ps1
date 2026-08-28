# monitor.ps1 — 康复LoRA流水线 实时监控窗口
# 放在 pipeline.py 同目录下运行；自动定位日志与产物目录。
# 用法:  powershell -NoExit -ExecutionPolicy Bypass -File monitor.ps1
# 显示内容:
#   1) 实时滚动 pipeline 日志(报错红 / 警告黄 / 成功绿)
#   2) 仪表盘: 运行状态 / 当前阶段 / 工程总量·剩余 / 当前接入的大模型 / 往来Token量 / 累计进度
# 说明: 仪表盘数据来自 output/run_status.json(pipeline 运行时同步写入), 缺失时退回 checkpoint.json

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

Write-Host "══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   康复LoRA流水线 实时监控" -ForegroundColor Cyan
Write-Host "   日志: $Log" -ForegroundColor DarkGray
Write-Host "   Ctrl+C 退出本窗口(不影响后台 pipeline)" -ForegroundColor DarkGray
Write-Host "══════════════════════════════════════════════════════════" -ForegroundColor Cyan

$lastCount = 0
$loop = 0

function Test-PipelineRunning {
    $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
             Where-Object { $_.CommandLine -like '*pipeline.py*' }
    return ($null -ne $procs)
}

function Bar($pct, $w = 26) {
    $p = [math]::Max(0, [math]::Min(100, $pct))
    $filled = [math]::Round($p / 100 * $w)
    return ('█' * $filled) + ('░' * ($w - $filled))
}

function Get-Status {
    # 优先 run_status.json(实时), 退回 checkpoint.json
    $rs = Join-Path $dir "output\run_status.json"
    $cp = Join-Path $dir "output\checkpoint.json"
    if (Test-Path $rs) {
        try { return (Get-Content $rs -Encoding utf8 | ConvertFrom-Json) } catch {}
    }
    if (Test-Path $cp) {
        try { return (Get-Content $cp -Encoding utf8 | ConvertFrom-Json) } catch {}
    }
    return $null
}

function Show-Dashboard {
    $st = Get-Status
    $running = Test-PipelineRunning
    $out = Join-Path $dir "output"
    $cleaned = 0; $qa = 0
    if (Test-Path (Join-Path $out "cleaned")) {
        $cleaned = (Get-ChildItem -Path (Join-Path $out "cleaned") -Recurse -Filter *.txt -ErrorAction SilentlyContinue).Count
    }
    if (Test-Path (Join-Path $out "qa_raw")) {
        $qa = (Get-ChildItem -Path (Join-Path $out "qa_raw") -Recurse -Filter *.json -ErrorAction SilentlyContinue).Count
    }
    $finalPath = Join-Path $out "final\rehab_lora_train_data.json"
    $final = Test-Path $finalPath
    $sample = 0
    if ($st -and $st.PSObject.Properties['sample_count']) { $sample = $st.sample_count }

    Write-Host ""
    if ($running) {
        Write-Host "● 状态: 运行中" -ForegroundColor Green
    } else {
        Write-Host "○ 状态: 未检测到 pipeline.py 进程 (可能已结束/未启动/崩溃)" -ForegroundColor Yellow
    }

    # 当前阶段
    $phase = if ($st -and $st.phase) { $st.phase } else { "—" }
    Write-Host ("  阶段     : {0}" -f $phase) -ForegroundColor White

    # 工程总量 / 剩余(本阶段)
    if ($st -and $st.total_tasks -gt 0) {
        $tot = $st.total_tasks
        $don = [math]::Min($st.done_tasks, $tot)
        $rem = $tot - $don
        $pct = if ($tot -gt 0) { $don / $tot * 100 } else { 0 }
        Write-Host ("  工程进度 : 总量 {0} | 已完成 {1} | 剩余 {2}  ({3:0.1f}%)" -f $tot, $don, $rem, $pct) -ForegroundColor White
        Write-Host ("            [{0}]" -f (Bar $pct)) -ForegroundColor Green
    } else {
        Write-Host "  工程进度 : 当前无活动子任务(或已结束)" -ForegroundColor DarkGray
    }

    # 当前接入的大模型
    $model = if ($st -and $st.current_model) { $st.current_model } else { "" }
    if ($model -eq "aixw") {
        Write-Host ("  当前模型 : aixw (主路) / {0}" -f $st.aixw_model) -ForegroundColor Magenta
    } elseif ($model -eq "scnet") {
        Write-Host ("  当前模型 : scnet (备路) / {0}" -f $st.scnet_model) -ForegroundColor Yellow
    } else {
        Write-Host "  当前模型 : — (尚无成功响应)" -ForegroundColor DarkGray
    }
    if ($st -and $st.aixw_fallback_count -gt 0) {
        Write-Host ("  ⚡ 主路→备路回退次数: {0}" -f $st.aixw_fallback_count) -ForegroundColor Red
    }

    # 往来 Token 量
    if ($st) {
        $ti = $st.total_input_tokens; $to = $st.total_output_tokens; $tc = $st.total_cache_tokens
        Write-Host ("  往来Token : 输入 {0:N0} | 输出 {1:N0} | 缓存命中 {2:N0}" -f $ti, $to, $tc) -ForegroundColor White
        Write-Host ("              aixw 输入/输出: {0:N0} / {1:N0}    scnet 输入/输出: {2:N0} / {3:N0}" `
                    -f $st.aixw_input_tokens, $st.aixw_output_tokens, $st.scnet_input_tokens, $st.scnet_output_tokens) `
                    -ForegroundColor DarkGray
    }

    # 累计进度
    $fin = if ($final) { "已生成 ($sample 条)" } else { "未生成" }
    Write-Host ("  累计     : 清洗 {0}块 | QA {1}块 | 最终产出 {2}" -f $cleaned, $qa, $fin) -ForegroundColor White

    # 配置信息
    if ($st -and $st.PSObject.Properties['workers']) {
        Write-Host ("  配置     : workers={0} | 备路启用={1}" -f $st.workers, $st.fallback_enabled) -ForegroundColor DarkGray
    }
    if ($st -and $st.last_update) {
        Write-Host ("  最后更新 : {0}" -f $st.last_update) -ForegroundColor DarkGray
    }
    Write-Host "──────────────────────────────────────────────────────────" -ForegroundColor Cyan
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
                } elseif ($l -match "⚠️|WARNING|重试|retry|网络异常|熔断") {
                    $fg = "Yellow"
                } elseif ($l -match "✅|完成|跳过|🎉|✔|阶段|🔧") {
                    $fg = "Green"
                }
                Write-Host $l -ForegroundColor $fg
            }
            $lastCount = $lines.Count
        }
    }
    $loop++
    if ($loop % 5 -eq 0) { Show-Dashboard }
    Start-Sleep -Seconds 1
}
