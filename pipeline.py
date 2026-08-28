import os
import sys
import json
import time
import subprocess
import shutil
import re
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# ============================================================
# 【配置区】改这里即可；所有路径默认相对"本脚本所在目录"，换电脑也能用
#  也可用环境变量覆盖(优先级高于此处)：
#   AIXW_API_KEY / REHAB_PDF_DIR / REHAB_OUTPUT_DIR / MINERU_CLI_PATH
# ============================================================
# 本脚本所在目录（自动获取，不写死盘符）
BASE_DIR = Path(__file__).resolve().parent

# ---- 极简 .env 加载器(无第三方依赖): 仅当对应环境变量未设置时, 从 .env 读取(不会覆盖已存在的环境变量) ----
def _load_env_file(path=".env"):
    try:
        with open(path, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line or _line.startswith("#") or "=" not in _line:
                    continue
                _k, _v = _line.split("=", 1)
                _k, _v = _k.strip(), _v.strip().strip('"').strip("'")
                if _k and _k not in os.environ:
                    os.environ[_k] = _v
    except FileNotFoundError:
        pass

_load_env_file()

# ===== LLM 服务商: 主路线 aixw (gpt-5.6-sol, Responses API) + 备用 scnet (DeepSeek-V4-Flash-0731, chat/completions) =====
# 主路线 aixw —— 关键: Responses API 端点 = {AIXW_BASE_URL}/responses (无 /v1), 认证用 Authorization: Bearer,
#   且必须带 User-Agent: OpenAI/Python (aixw 网关据此放行上游, 缺此头一律 Upstream access forbidden)
AIXW_API_KEY = os.environ.get("AIXW_API_KEY", "")
AIXW_MODEL = os.environ.get("AIXW_MODEL", "gpt-5.6-sol")
AIXW_BASE_URL = "https://api.aixw.org"
# 备用 scnet —— 标准 OpenAI chat/completions
SCNET_API_KEY = os.environ.get("SCNET_API_KEY", "")
SCNET_MODEL = os.environ.get("SCNET_MODEL", "DeepSeek-V4-Flash-0731")
SCNET_BASE_URL = "https://api.scnet.cn/api/llm/v1"
# 失败自动回退: aixw 限流/5xx/超时 -> 自动转 scnet; 设 REHAB_DISABLE_FALLBACK=1 可强制只用 aixw
FALLBACK_ENABLED = os.environ.get("REHAB_DISABLE_FALLBACK", "").lower() not in ("1", "true", "yes")
# 并发 worker 数(提速核心): 默认 12, REHAB_WORKERS 覆盖
WORKERS = int(os.environ.get("REHAB_WORKERS", "12") or "12")
# aixw 单块重试次数(用尽即转备用); 设小以快速失败避免拖慢整体
AIXW_MAX_RETRIES = int(os.environ.get("REHAB_AIXW_RETRIES", "2") or "2")
# 计费单价(元 / 百万 token)—— 未知先留 0(费用估算显示¥0)
PRICE_INPUT = 0.0
PRICE_OUTPUT = 0.0
PRICE_CACHE = 0.0
PDF_INPUT_DIR = os.environ.get("REHAB_PDF_DIR", str(BASE_DIR / "pdf"))      # 放 PDF 的子目录
OUTPUT_ROOT = os.environ.get("REHAB_OUTPUT_DIR", str(BASE_DIR / "output"))  # 所有产出都在这里
MINERU_CLI_PATH = os.environ.get("MINERU_CLI_PATH", "")  # 留空则自动探测(PATH/常见conda环境)
MINERU_BACKEND = "pipeline"
DEFAULT_LANG = "ch"
# 阶段3每本书生成Q&A条数上限: 0=不限制(全量覆盖全书所有清洗块); 设正整数则达到即停
#   也可用环境变量 REHAB_QA_MAX 覆盖(优先级高于此处)
QA_MAX_PER_BOOK = int(os.environ.get("REHAB_QA_MAX", "0") or "0")
# 全本重跑模式: 设 REHAB_FULL_BOOK=1 时, 即使 qa_done 已标记该书, 阶段3仍重进本书
#   (已生成的块靠块级 qa_chunks_done 跳过, 不会重复生成 → 最终自然合并之前demo的成果)
FULL_BOOK = os.environ.get("REHAB_FULL_BOOK", "").lower() in ("1", "true", "yes")
# 阶段2清洗块数上限: 0=清洗全书所有块(慢); 设正整数则只洗前N块, 足够阶段3采样即可(适合快速验证/demo)
#   例如设 50 → 阶段2洗50块后停止, 阶段3从其中采样约38块生成Q&A, 几分钟出成果
#   也可用环境变量 REHAB_CLEAN_MAX_BLOCKS 覆盖(优先级高于此处)
CLEAN_MAX_BLOCKS = int(os.environ.get("REHAB_CLEAN_MAX_BLOCKS", "0") or "0")
API_DELAY = 0  # 并发模式下不再逐块sleep; 保留变量兼容旧逻辑
CHECKPOINT_FILE = ""

# ============================================================
# 【初始化】
# ============================================================
if not CHECKPOINT_FILE:
    CHECKPOINT_FILE = os.path.join(OUTPUT_ROOT, "checkpoint.json")

for sub in ["markdown", "cleaned", "qa_raw", "final", "logs"]:
    os.makedirs(os.path.join(OUTPUT_ROOT, sub), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(
            os.path.join(OUTPUT_ROOT, "logs", "pipeline.log"), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# 【断点续传】
# ============================================================
def load_checkpoint() -> Dict:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "mineru_done": [], "clean_done": [], "qa_done": [],
        "total_input_tokens": 0, "total_output_tokens": 0,
        "total_cache_tokens": 0, "last_update": ""
    }

def save_checkpoint(cp: Dict):
    cp["last_update"] = datetime.now().isoformat()
    tmp = CHECKPOINT_FILE + ".tmp"
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(cp, f, ensure_ascii=False, indent=2)
    os.replace(tmp, CHECKPOINT_FILE)

def estimate_cost(cp: Dict) -> float:
    return round(
        cp["total_input_tokens"] * PRICE_INPUT / 1_000_000 +
        cp["total_output_tokens"] * PRICE_OUTPUT / 1_000_000 +
        cp["total_cache_tokens"] * PRICE_CACHE / 1_000_000, 4)

# ============================================================
# 【LLM API调用】主路线 aixw (Responses API) + 备用 scnet (chat/completions), 含并发安全与熔断
# ============================================================
_CP_LOCK = threading.Lock()        # 保护 cp 的 token 统计与 checkpoint 写入
_AIXW_STATE = {"fail_streak": 0, "disabled": False}
_AIXW_LOCK = threading.Lock()

def _acct(cp, usage):
    """累加 token 用量(线程安全)。兼容 chat/completions 与 responses 两种 usage 字段名。"""
    if not usage:
        return
    with _CP_LOCK:
        inp = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        out = usage.get("completion_tokens") or usage.get("output_tokens") or 0
        cp["total_input_tokens"] += inp
        cp["total_output_tokens"] += out
        details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
        cache_hit = usage.get("prompt_cache_hit_tokens", 0) or details.get("cached_tokens", 0)
        cp["total_cache_tokens"] += cache_hit

def _post(url, headers, payload, timeout):
    req = Request(url, data=payload, headers=headers, method='POST')
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode('utf-8')

def _parse_aixw_output(result):
    """Responses API 返回 output_text 或 output[].content[].text。"""
    if not isinstance(result, dict):
        return None
    text = result.get("output_text")
    if text:
        return text
    for item in result.get("output", []) or []:
        for c in item.get("content", []) or []:
            if isinstance(c, dict) and c.get("type") == "output_text":
                return c.get("text", "")
    return None

def _read_aixw_stream(resp):
    """解析 Responses API 的 SSE 流式响应, 拼接 output_text 增量; 返回 (text, usage|None)。
    流式调用规避官方对非流式请求的限流。"""
    buf, usage = [], None
    for raw_line in resp:
        line = raw_line.decode('utf-8', 'replace').strip()
        if not line or line.startswith(':'):
            continue  # 空行 / SSE 心跳注释
        if not line.startswith('data:'):
            continue
        data = line[len('data:'):].strip()
        if data == '[DONE]':
            break
        try:
            evt = json.loads(data)
        except json.JSONDecodeError:
            continue
        t = evt.get('type')
        # 文本增量(主路径): response.output_text.delta 事件的 delta 字段
        delta = evt.get('delta')
        if delta:
            buf.append(delta)
        # 部分实现在 response.output_text 事件中给累计/全文 text(仅当无 delta 时采用, 防重复)
        elif t == 'response.output_text' and evt.get('text'):
            buf.append(evt['text'])
        # 用量: 出现在 response.completed / response.usage.updated 事件
        if evt.get('usage'):
            usage = evt['usage']
        # 错误 / 失败事件
        if t == 'error' or evt.get('error'):
            raise RuntimeError(f"aixw 流式错误: {evt.get('error') or evt}")
        if t == 'response.failed':
            raise RuntimeError(f"aixw 流式失败: {evt.get('status')}")
    return ''.join(buf), usage


def _call_aixw(messages, cp, temperature, max_tokens):
    """主路线(流式)。成功返回文本, 失败返回 None(交由调用方回退 scnet)。绝不 sys.exit。"""
    sys_parts = [m["content"] for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    if not rest:
        rest = [{"role": "user", "content": messages[-1].get("content", "")}]
    body = {
        "model": AIXW_MODEL,
        "input": rest,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 0.95,
        "stream": True,                     # 流式: 规避官方对非流式调用的限流
    }
    if sys_parts:
        body["instructions"] = "\n".join(sys_parts)
    payload = json.dumps(body).encode('utf-8')
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Authorization": f"Bearer {AIXW_API_KEY}",
        "User-Agent": "OpenAI/Python 3.5.0",   # aixw 网关按此 UA 放行上游, 缺则 forbidden
    }
    url = f"{AIXW_BASE_URL}/responses"
    for attempt in range(AIXW_MAX_RETRIES):
        try:
            req = Request(url, data=payload, headers=headers, method='POST')
            with urlopen(req, timeout=180) as resp:
                text, usage = _read_aixw_stream(resp)
        except HTTPError as e:
            b = e.read().decode('utf-8', 'replace') if e.fp else ""
            if e.code == 429:
                wait = (attempt + 1) * 4
                logger.warning(f"  ⚠️ aixw 限流, {wait}s后重试 ({attempt+1}/{AIXW_MAX_RETRIES})")
                time.sleep(wait); continue
            logger.warning(f"  ⚠️ aixw HTTP {e.code}: {b[:160]}")
            return None
        except (URLError, TimeoutError) as e:
            logger.warning(f"  ⚠️ aixw 网络异常: {e}, 重试 ({attempt+1}/{AIXW_MAX_RETRIES})")
            time.sleep(3); continue
        except Exception as e:
            logger.warning(f"  ⚠️ aixw 异常: {e}, 重试")
            time.sleep(3); continue
        if not text or not text.strip():
            logger.warning(f"  ⚠️ aixw 空响应, 重试 ({attempt+1}/{AIXW_MAX_RETRIES})")
            time.sleep(2); continue
        if usage:
            _acct(cp, usage)
        return text
    return None

def _call_scnet(messages, cp, temperature, max_tokens):
    """备用路线。成功返回文本, 失败返回 None。"""
    url = f"{SCNET_BASE_URL}/chat/completions"
    payload = json.dumps({
        "model": SCNET_MODEL, "messages": messages,
        "temperature": temperature, "max_tokens": max_tokens, "top_p": 0.95,
    }).encode('utf-8')
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {SCNET_API_KEY}",
    }
    for attempt in range(3):
        try:
            raw = _post(url, headers, payload, timeout=180)
        except HTTPError as e:
            b = e.read().decode('utf-8', 'replace') if e.fp else ""
            if e.code == 429:
                wait = (attempt + 1) * 5
                logger.warning(f"  ⚠️ scnet 限流, {wait}s后重试 ({attempt+1}/3)")
                time.sleep(wait); continue
            if e.code == 401:
                logger.error("  ❌ scnet API Key 无效!"); return None
            logger.warning(f"  ⚠️ scnet HTTP {e.code}: {b[:160]}, 重试")
            time.sleep(8); continue
        except (URLError, TimeoutError) as e:
            logger.warning(f"  ⚠️ scnet 网络异常: {e}, 重试")
            time.sleep(3); continue
        except Exception as e:
            logger.warning(f"  ⚠️ scnet 异常: {e}, 重试")
            time.sleep(3); continue
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"  ⚠️ scnet 非JSON响应, 重试")
            time.sleep(3); continue
        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            logger.warning(f"  ⚠️ scnet 响应结构异常: {str(result)[:120]}, 重试")
            time.sleep(3); continue
        if not content or not content.strip():
            logger.warning(f"  ⚠️ scnet 空响应, 重试")
            time.sleep(2); continue
        _acct(cp, result.get("usage"))
        return content
    return None

def call_llm(messages: List[Dict], cp: Dict, temperature: float = 0.3,
             max_tokens: int = 8192) -> Optional[str]:
    """统一入口: 主 aixw, 失败(或熔断)回退 scnet。含 aixw 熔断(连续失败即整体切备用)。"""
    with _AIXW_LOCK:
        aixw_ok = FALLBACK_ENABLED and not _AIXW_STATE["disabled"]
    if aixw_ok:
        content = _call_aixw(messages, cp, temperature, max_tokens)
        if content is not None:
            with _AIXW_LOCK:
                _AIXW_STATE["fail_streak"] = 0
            return content
        with _AIXW_LOCK:
            _AIXW_STATE["fail_streak"] += 1
            if _AIXW_STATE["fail_streak"] >= 5:
                _AIXW_STATE["disabled"] = True
                logger.warning("  ⚡ aixw 连续失败5次 → 本批次剩余请求直接走 scnet(熔断)")
    content = _call_scnet(messages, cp, temperature, max_tokens)
    return content

# ============================================================
# 【文本分块】
# ============================================================
def split_into_chunks(text: str, max_chars: int = 3000) -> List[str]:
    chunks, current = [], ""
    for line in text.split('\n'):
        if line.startswith('#') and len(current) > 200:
            chunks.append(current.strip())
            current = line
        else:
            current += '\n' + line
            if len(current) > max_chars:
                chunks.append(current.strip())
                current = ""
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if len(c) > 50]

# ============================================================
# 【阶段1】MinerU批量PDF->Markdown
# ============================================================
def get_mineru_cmd():
    # 1) 显式配置(配置区或环境变量)
    if MINERU_CLI_PATH and os.path.exists(MINERU_CLI_PATH):
        return MINERU_CLI_PATH
    # 2) 系统 PATH
    which = shutil.which("mineru")
    if which:
        return which
    # 3) 常见 conda 安装位置（覆盖多平台）
    candidates = [
        os.path.expanduser(r"~/miniconda3/envs/mineru/Scripts/mineru.exe"),
        os.path.expanduser(r"~/anaconda3/envs/mineru/Scripts/mineru.exe"),
        r"C:\Users\johnz\miniconda3\envs\mineru\Scripts\mineru.exe",
        os.path.expanduser(r"~/miniconda3/envs/mineru/bin/mineru"),
        os.path.expanduser(r"~/anaconda3/envs/mineru/bin/mineru"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # 4) 兜底，让 subprocess 自己报"找不到"并给出友好提示
    return "mineru"

def detect_lang(filename: str) -> Optional[str]:
    # MinerU 的 -l 枚举没有 'en'；含中文的书显式给 'ch'，否则不传 -l(自动识别)
    if re.search(r'[\u4e00-\u9fff]', filename):
        return DEFAULT_LANG
    return None

def phase1_mineru(cp: Dict):
    sep = "=" * 60
    logger.info(f"\n{sep}\n【阶段1】MinerU 批量PDF->Markdown\n{sep}")
    pdf_dir = Path(PDF_INPUT_DIR)
    if not pdf_dir.exists():
        logger.error(f"PDF目录不存在: {PDF_INPUT_DIR}")
        sys.exit(1)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    logger.info(f"  找到 {len(pdfs)} 本PDF")
    mineru_cmd = get_mineru_cmd()
    logger.info(f"  MinerU命令: {mineru_cmd}")
    for i, pdf in enumerate(pdfs, 1):
        book = pdf.stem
        if book in cp["mineru_done"]:
            logger.info(f"  [{i}/{len(pdfs)}] ⏭️ 跳过: {book}")
            continue
        lang = detect_lang(pdf.name)
        out_dir = os.path.join(OUTPUT_ROOT, "markdown", book)
        os.makedirs(out_dir, exist_ok=True)
        cmd = [mineru_cmd, "-p", str(pdf), "-o", out_dir,
               "-b", MINERU_BACKEND, "-m", "auto"]
        if lang:  # 仅中文书传 -l ch；英文书留空让 MinerU 自动
            cmd += ["-l", lang]
        logger.info(f"  [{i}/{len(pdfs)}] 📖 {book} ({'EN' if lang=='en' else 'CN'})")
        try:
            ret = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=3600, encoding='utf-8', errors='replace')
            if ret.returncode == 0:
                cp["mineru_done"].append(book)
                save_checkpoint(cp)
                logger.info(f"           ✅ 完成")
            else:
                logger.error(f"           ❌ 失败: {ret.stderr[:300]}")
        except FileNotFoundError:
            logger.error(f"           ❌ 找不到mineru命令! 请先运行 rely.py")
            sys.exit(1)
        except subprocess.TimeoutExpired:
            logger.error(f"           ⏰ 超时(>1h)")
        except Exception as e:
            logger.error(f"           ❌ 异常: {e}")
        time.sleep(2)
    logger.info(f"  阶段1完成: {len(cp['mineru_done'])}/{len(pdfs)} 本")

# ============================================================
# 【阶段2】文本清洗
# ============================================================
SYSTEM_PROMPT_CLEAN = (
    "你是康复医学文本编辑。清洗OCR扫描文本错误, 严格遵守:\n"
    "1.修正OCR错字(康发->康复,患看->患者)\n"
    "2.合并断行句子\n"
    "3.去除页码/页眉/页脚\n"
    "4.保留专业术语/数据/公式原样\n"
    "5.保留Markdown格式\n"
    "6.不添加原文没有的内容\n"
    "7.只输出清洗后文本"
)

def phase2_clean(cp: Dict):
    sep = "=" * 60
    logger.info(f"\n{sep}\n【阶段2】LLM文本清洗 (主:aixw {AIXW_MODEL} / 备:scnet {SCNET_MODEL})\n{sep}")
    md_dir = Path(OUTPUT_ROOT) / "markdown"
    cleaned_dir = Path(OUTPUT_ROOT) / "cleaned"
    for book_dir in sorted(md_dir.iterdir()):
        if not book_dir.is_dir():
            continue
        book = book_dir.name
        md_files = list(book_dir.rglob("*.md"))  # 递归：兼容 MinerU 的嵌套输出目录
        if not md_files:
            logger.warning(f"  ⚠️ {book} 无.md文件")
            continue
        full_md = "\n\n".join(f.read_text(encoding='utf-8') for f in md_files)
        chunks = split_into_chunks(full_md)
        logger.info(f"  📖 {book}: {len(chunks)} 块")
        # 快速验证: 已清洗块数已达上限, 直接跳过阶段2(避免全书逐块调API)
        if CLEAN_MAX_BLOCKS and len(cp["clean_done"]) >= CLEAN_MAX_BLOCKS:
            logger.info(f"  🔼 已清洗 {len(cp['clean_done'])}块 >= CLEAN_MAX_BLOCKS={CLEAN_MAX_BLOCKS}, 跳过阶段2清洗")
            continue
        # 收集已完成的块(跳过), 其余入并发任务
        cleaned_map = {}
        tasks = []
        for j, chunk in enumerate(chunks):
            cid = f"{book}_c{j}"
            chunk_file = cleaned_dir / book / f"c{j:04d}.txt"
            if cid in cp["clean_done"] and chunk_file.exists():
                cleaned_map[j] = chunk_file.read_text(encoding='utf-8')
                if CLEAN_MAX_BLOCKS and len(cleaned_map) >= CLEAN_MAX_BLOCKS:
                    break
                continue
            if CLEAN_MAX_BLOCKS and len(cleaned_map) + len(tasks) >= CLEAN_MAX_BLOCKS:
                break
            tasks.append((j, chunk))
        logger.info(f"  🔧 并发清洗: 已完成{len(cleaned_map)}块, 待处理{len(tasks)}块, workers={WORKERS}")

        def _clean_worker(job):
            j, chunk = job
            resp = call_llm([
                {"role": "system", "content": SYSTEM_PROMPT_CLEAN},
                {"role": "user", "content": f"清洗以下OCR文本:\n\n{chunk}"}
            ], cp, temperature=0.3)
            text = resp if resp else chunk
            cf = cleaned_dir / book / f"c{j:04d}.txt"
            cf.parent.mkdir(parents=True, exist_ok=True)
            cf.write_text(text, encoding='utf-8')
            with _CP_LOCK:
                if f"{book}_c{j}" not in cp["clean_done"]:
                    cp["clean_done"].append(f"{book}_c{j}")
            return j, text

        new_map = {}
        if tasks:
            done = 0
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                futs = {ex.submit(_clean_worker, job): job[0] for job in tasks}
                for fut in as_completed(futs):
                    j, text = fut.result()
                    new_map[j] = text
                    done += 1
                    if done % 20 == 0 or done == len(tasks):
                        with _CP_LOCK:
                            save_checkpoint(cp)
                        logger.info(f"    🔧 清洗 {done}/{len(tasks)}")
        # 按顺序合并(已跳过 + 新洗)
        ordered = []
        for j in range(len(chunks)):
            if j in cleaned_map:
                ordered.append(cleaned_map[j])
            elif j in new_map:
                ordered.append(new_map[j])
        merged = "\n\n---\n\n".join(ordered)
        (cleaned_dir / f"{book}_cleaned.md").write_text(merged, encoding='utf-8')
        with _CP_LOCK:
            save_checkpoint(cp)
        logger.info(f"  ✅ {book} 清洗完成 ({len(merged)}字, 累计{len(cp['clean_done'])}块)")
    logger.info(f"  阶段2完成 | 累计费用: ¥{estimate_cost(cp)}")

# ============================================================
# 【阶段3】生成Q&A(缓存优化)
# ============================================================
SYSTEM_PROMPT_QA = (
    "你是康复医学教育专家, 为公益康复指导App准备训练数据。\n"
    "根据提供的教材内容生成高质量问答对。\n\n"
    "要求:\n"
    "1.问题像真实患者/康复师的自然提问\n"
    "2.答案严格基于文本, 不得编造\n"
    "3.答案专业准确有条理\n"
    "4.操作类必须含注意事项和禁忌症\n"
    "5.表格数据转为问答\n"
    "6.标注来源章节\n"
    "7.输出JSON数组:[{\"question\":\"...\",\"answer\":\"...\",\"source\":\"...\"}]"
)

def phase3_qa(cp: Dict):
    sep = "=" * 60
    logger.info(f"\n{sep}\n【阶段3】LLM生成Q&A (主:aixw {AIXW_MODEL} / 备:scnet {SCNET_MODEL})\n{sep}")
    cleaned_dir = Path(OUTPUT_ROOT) / "cleaned"
    qa_dir = Path(OUTPUT_ROOT) / "qa_raw"
    cp.setdefault("qa_chunks_done", [])
    # 直接读阶段2产出的零散块文件 cleaned/<book>/c*.txt (不再依赖合并后的 _cleaned.md,
    # 这样阶段2即使只洗了部分块, 阶段3也能立即开工)
    for book_dir in sorted(d for d in cleaned_dir.iterdir() if d.is_dir()):
        book = book_dir.name
        if book in cp["qa_done"]:
            if FULL_BOOK:
                logger.info(f"  🔄 全本模式: 重进 {book} (已生成块靠qa_chunks_done跳过)")
            else:
                logger.info(f"  ⏭️ 跳过: {book}")
                continue
        chunk_files = sorted(book_dir.glob("c*.txt"))
        if not chunk_files:
            # 兜底: 若只有合并文件, 也支持
            merged = cleaned_dir / f"{book}_cleaned.md"
            if merged.exists():
                chunk_files = [merged]
            else:
                logger.warning(f"  ⚠️ {book} 无清洗块文件(c*.txt), 跳过")
                continue
        # 拼接已清洗块, 切成适合QA的片段
        text = "\n\n---\n\n".join(f.read_text(encoding='utf-8') for f in chunk_files)
        chunks = split_into_chunks(text, max_chars=2000)
        # 全量模式: 处理全书所有清洗块(不再采样前38块); 已生成块靠 qa_chunks_done 跳过
        to_process = chunks
        logger.info(f"  📖 {book}: 可用{len(chunk_files)}块, 处理{len(to_process)}块 (全量)")
        chunk_dir = qa_dir / book
        chunk_dir.mkdir(parents=True, exist_ok=True)
        # 复用已生成的块(断点续传)
        reused = []
        tasks = []
        done_cids = set(cp["qa_chunks_done"])
        for j, chunk in enumerate(to_process):
            cid = f"{book}_q{j}"
            cfile = chunk_dir / f"q{j:04d}.json"
            if cid in done_cids and cfile.exists():
                try:
                    reused.extend(json.loads(cfile.read_text(encoding='utf-8')))
                except Exception:
                    pass
                if QA_MAX_PER_BOOK and len(reused) >= QA_MAX_PER_BOOK:
                    break
                continue
            if QA_MAX_PER_BOOK and len(reused) >= QA_MAX_PER_BOOK:
                break
            tasks.append((j, cid, chunk))
        logger.info(f"  🔧 并发QA: 复用{len(reused)}条, 待处理{len(tasks)}块, workers={WORKERS}")

        def _qa_worker(job):
            j, cid, chunk = job
            resp = call_llm([
                {"role": "system", "content": SYSTEM_PROMPT_QA},
                {"role": "user", "content":
                 f"以下是《{book}》的内容片段:\n\n---\n{chunk}\n---\n\n"
                 f"生成3~5个康复医学问答对, JSON数组格式。"}
            ], cp, temperature=0.5)
            items = []
            if resp:
                m = re.search(r'\[.*\]', resp, re.DOTALL)
                if m:
                    try:
                        raw = json.loads(m.group())
                        for item in raw:
                            if isinstance(item, dict) and "question" in item and "answer" in item:
                                items.append({"question": item["question"],
                                              "answer": item["answer"],
                                              "source": item.get("source", book)})
                    except json.JSONDecodeError:
                        logger.warning(f"    ⚠️ 块{j} JSON解析失败")
            cfile = chunk_dir / f"q{j:04d}.json"
            if items:
                cfile.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding='utf-8')
                with _CP_LOCK:
                    if cid not in cp["qa_chunks_done"]:
                        cp["qa_chunks_done"].append(cid)
            return items

        new_items = []
        if tasks:
            done = 0
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                futs = [ex.submit(_qa_worker, job) for job in tasks]
                for fut in as_completed(futs):
                    new_items.extend(fut.result())
                    done += 1
                    if done % 20 == 0 or done == len(tasks):
                        with _CP_LOCK:
                            save_checkpoint(cp)
                        logger.info(f"    🔧 QA {done}/{len(tasks)} (累计{len(reused)+len(new_items)}条)")
        all_qa = reused + new_items
        if QA_MAX_PER_BOOK:
            all_qa = all_qa[:QA_MAX_PER_BOOK]
        qa_file = qa_dir / f"{book}_qa.json"
        with open(qa_file, 'w', encoding='utf-8') as f:
            json.dump(all_qa, f, ensure_ascii=False, indent=2)
        with _CP_LOCK:
            if book not in cp["qa_done"]:
                cp["qa_done"].append(book)
            save_checkpoint(cp)
        logger.info(f"  ✅ {book}: {len(all_qa)}条Q&A")
    logger.info(f"  阶段3完成 | 累计费用: ¥{estimate_cost(cp)}")

# ============================================================
# 【阶段4】导出LLaMA-Factory ShareGPT格式
# ============================================================
def phase4_export(cp: Dict):
    sep = "=" * 60
    logger.info(f"\n{sep}\n【阶段4】导出LLaMA-Factory格式\n{sep}")
    qa_dir = Path(OUTPUT_ROOT) / "qa_raw"
    final_dir = Path(OUTPUT_ROOT) / "final"
    conversations = []
    for qf in sorted(qa_dir.glob("*_qa.json")):
        with open(qf, 'r', encoding='utf-8') as f:
            qas = json.load(f)
        book = qf.stem.replace("_qa", "")
        logger.info(f"  📖 {book}: {len(qas)}条")
        for qa in qas:
            conversations.append({
                "conversations": [
                    {"from": "human", "value": qa["question"]},
                    {"from": "gpt", "value": qa["answer"]}
                ]
            })
    out_json = final_dir / "rehab_lora_train_data.json"
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)
    dataset_info = {
        "rehab_qa": {
            "file_name": "rehab_lora_train_data.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
            "tags": {
                "role_tag": "from", "content_tag": "value",
                "user_tag": "human", "assistant_tag": "gpt"
            }
        }
    }
    info_file = final_dir / "dataset_info.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    logger.info(f"  ✅ 训练数据: {out_json}")
    logger.info(f"  ✅ 数据集配置: {info_file}")
    logger.info(f"  📊 总样本数: {len(conversations)}")
    logger.info(f"  📋 将 {final_dir} 上传到阿里云LLaMA-Factory即可训练")

# ============================================================
# 【主入口】
# ============================================================
def main():
    logger.info("=" * 60)
    logger.info("  康复医学书籍 -> LoRA训练数据 自动化流水线")
    logger.info("=" * 60)
    if not AIXW_API_KEY or not AIXW_API_KEY.strip():
        logger.error("❌ 请先填入 aixw API Key! (环境变量 AIXW_API_KEY)")
    if not SCNET_API_KEY or not SCNET_API_KEY.strip():
        logger.error("❌ 请先填入 scnet API Key! (环境变量 SCNET_API_KEY)")
    if PRICE_INPUT == 0 and PRICE_OUTPUT == 0:
        logger.warning("⚠️ 未配置计费单价(PRICE_INPUT/PRICE_OUTPUT)，费用估算将显示 ¥0；请在配置区填入 SCNet DeepSeek-V4-Flash-0731 实际单价")
    cp = load_checkpoint()
    if cp["last_update"]:
        logger.info(f"📌 续传模式 | 上次: {cp['last_update']}")
        logger.info(f"   MinerU:{len(cp['mineru_done'])}本 | "
                     f"清洗:{len(cp['clean_done'])}块 | "
                     f"Q&A:{len(cp['qa_done'])}本 | "
                     f"费用:¥{estimate_cost(cp)}")
    t0 = time.time()
    try:
        phase1_mineru(cp)
        phase2_clean(cp)
        phase3_qa(cp)
        phase4_export(cp)
    except Exception as _fatal:
        logger.error("=" * 60)
        logger.error(f"❌ 流水线异常终止: {type(_fatal).__name__}: {_fatal}")
        import traceback as _tb
        for _line in _tb.format_exc().splitlines():
            logger.error(f"   {_line}")
        logger.error("=" * 60)
        logger.error("已保留断点(checkpoint.json), 修复后重跑可续传。")
        raise
    elapsed = (time.time() - t0) / 60
    cost = estimate_cost(cp)
    logger.info(f"\n{'='*60}")
    logger.info(f"🎉 全部完成!")
    logger.info(f"{'='*60}")
    logger.info(f"  耗时: {elapsed:.1f}分钟")
    logger.info(f"  输入Token: {cp['total_input_tokens']:,}")
    logger.info(f"  输出Token: {cp['total_output_tokens']:,}")
    logger.info(f"  缓存命中: {cp['total_cache_tokens']:,}")
    logger.info(f"  总费用: ¥{cost}")
    logger.info(f"  输出: {OUTPUT_ROOT}/final/rehab_lora_train_data.json")
    logger.info(f"  ⚠️ 安全提醒:")
    logger.info(f"  - 请人工抽检>=10%的Q&A确保医学准确性")
    logger.info(f"  - 康复指导涉及患者安全, 务必经专业医师审核")
    logger.info(f"  - 建议先用小批量数据试训验证效果")

if __name__ == "__main__":
    # 可选: 弹出 PowerShell 实时监控窗口 (python pipeline.py --monitor)
    # 也可用环境变量 REHAB_MONITOR=1 开启。窗口独立于主流程, 关掉不影响流水线。
    _args = sys.argv[1:]
    if "--monitor" in _args or os.environ.get("REHAB_MONITOR") == "1":
        _monitor = os.path.join(BASE_DIR, "monitor.ps1")
        if os.path.exists(_monitor):
            try:
                import subprocess
                subprocess.Popen(
                    ["powershell.exe", "-NoExit", "-ExecutionPolicy", "Bypass",
                     "-File", _monitor],
                    creationflags=0x00000008,  # CREATE_NEW_CONSOLE: 弹独立窗口
                )
                print("🖥️ 已弹出 PowerShell 监控窗口 (实时进度/报错)")
            except Exception as _e:
                print(f"⚠️ 监控窗口启动失败(不影响主流程): {_e}")
        else:
            print(f"⚠️ 未找到 monitor.ps1 ({_monitor})，跳过监控窗口")
    main()