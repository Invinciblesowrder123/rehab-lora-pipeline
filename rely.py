import os
import sys
import json
import subprocess
import shutil
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime

# ============================================================
# 【配置区】
# ============================================================

# 智谱API Key（用于测试连通性，填不填都行，不填则跳过API测试）
ZHIPU_API_KEY = ""  # ← 可选：填入后会自动测试API连通性

# Conda环境名称
CONDA_ENV_NAME = "mineru"

# Conda中Python版本
CONDA_PYTHON_VERSION = "3.10"

# MinerU安装命令
MINERU_INSTALL_CMD = 'pip install -U "mineru[all]" -i https://mirrors.aliyun.com/pypi/simple'

# 智谱API测试地址
ZHIPU_TEST_URL = "https://open.bigmodel.cn/api/paas/v4/models"

# ============================================================
# 【颜色输出工具】（Windows兼容）
# ============================================================

def _enable_ansi():
    """Windows终端启用ANSI颜色"""
    if sys.platform == 'win32':
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass

_enable_ansi()

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


def ok(msg):   print(f"  {GREEN}✅{RESET} {msg}")
def fail(msg): print(f"  {RED}❌{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}⚠️{RESET}  {msg}")
def info(msg): print(f"  {CYAN}ℹ️{RESET}  {msg}")
def header(msg): print(f"\n{BOLD}{'='*60}\n  {msg}\n{'='*60}{RESET}")


# ============================================================
# 【检查函数】
# ============================================================

class EnvChecker:
    def __init__(self):
        self.results = {}
        self.mineru_path = None
        self.conda_path = None

    # ---------- 1. Python版本 ----------
    def check_python(self):
        header("1/6 检查Python版本")
        ver = sys.version_info
        ver_str = f"{ver.major}.{ver.minor}.{ver.micro}"
        info(f"当前Python: {ver_str} ({sys.executable})")

        if ver.major == 3 and ver.minor >= 9:
            ok(f"Python {ver_str} 满足要求 (≥3.9)")
            self.results["python"] = True
        else:
            fail(f"Python {ver_str} 版本过低，需要 ≥3.9")
            self.results["python"] = False

    # ---------- 2. Conda ----------
    def check_conda(self):
        header("2/6 检查Conda环境")

        # 查找conda
        conda_cmd = shutil.which("conda")
        if not conda_cmd:
            # 尝试常见路径
            common_paths = [
                os.path.expanduser("~/miniconda3/Scripts/conda.exe"),
                os.path.expanduser("~/anaconda3/Scripts/conda.exe"),
                r"C:\ProgramData\miniconda3\Scripts\conda.exe",
                r"C:\ProgramData\anaconda3\Scripts\conda.exe",
            ]
            for p in common_paths:
                if os.path.exists(p):
                    conda_cmd = p
                    break

        if conda_cmd:
            self.conda_path = conda_cmd
            ok(f"找到Conda: {conda_cmd}")
            self.results["conda"] = True

            # 检查mineru环境是否存在
            try:
                result = subprocess.run(
                    [conda_cmd, "env", "list", "--json"],
                    capture_output=True, text=True, timeout=30
                )
                envs = json.loads(result.stdout).get("envs", [])
                env_names = [os.path.basename(e) for e in envs]

                if CONDA_ENV_NAME in env_names:
                    ok(f"Conda环境 '{CONDA_ENV_NAME}' 已存在")
                    self.results["conda_env"] = True
                else:
                    warn(f"Conda环境 '{CONDA_ENV_NAME}' 不存在，将自动创建")
                    self.results["conda_env"] = False
            except Exception as e:
                warn(f"无法列出Conda环境: {e}")
                self.results["conda_env"] = False
        else:
            fail("未找到Conda！请先安装 Miniconda:")
            fail("  下载地址: https://docs.conda.io/en/latest/miniconda.html")
            self.results["conda"] = False
            self.results["conda_env"] = False

    # ---------- 3. MinerU安装 ----------
    def check_mineru(self):
        header("3/6 检查MinerU安装")

        if not self.results.get("conda"):
            fail("Conda不可用，跳过MinerU检查")
            self.results["mineru"] = False
            return

        # 在conda环境中检查mineru
        try:
            result = subprocess.run(
                [self.conda_path, "run", "-n", CONDA_ENV_NAME,
                 "python", "-c", "import importlib.metadata as _m; print(_m.version('mineru'))"],
                capture_output=True, text=True, timeout=30
            )

            if result.returncode == 0:
                version = result.stdout.strip().split('\n')[-1]
                ok(f"MinerU已安装，版本: {version}")
                self.results["mineru"] = True

                # 查找mineru CLI路径
                cli_result = subprocess.run(
                    [self.conda_path, "run", "-n", CONDA_ENV_NAME,
                     "where" if sys.platform == 'win32' else "which", "mineru"],
                    capture_output=True, text=True, timeout=10
                )
                if cli_result.returncode == 0:
                    self.mineru_path = cli_result.stdout.strip().split('\n')[0]
                    info(f"MinerU CLI路径: {self.mineru_path}")
            else:
                warn("MinerU未安装或导入失败")
                self.results["mineru"] = False
        except Exception as e:
            warn(f"检查MinerU异常: {e}")
            self.results["mineru"] = False

    # ---------- 4. MinerU模型权重 ----------
    def check_mineru_models(self):
        header("4/6 检查MinerU模型权重")

        if not self.results.get("mineru"):
            fail("MinerU未安装，跳过模型检查")
            self.results["mineru_models"] = False
            return

        # MinerU权重存放位置随版本变化：v1在 ~/.cache/mineru(.pt/.onnx)，
        # v2/v3改存于 ~/.cache/modelscope 或 ~/.cache/huggingface(.safetensors)
        model_dirs = [
            os.path.expanduser("~/.cache/mineru"),
            os.path.expanduser("~/.cache/modelscope"),
            os.path.expanduser("~/.cache/huggingface"),
            os.path.expanduser("~/AppData/Local/mineru"),
            os.environ.get("MINERU_MODEL_DIR", ""),
        ]

        found = False
        for d in model_dirs:
            if d and os.path.exists(d):
                files = (list(Path(d).rglob("*.pt"))
                         + list(Path(d).rglob("*.pth"))
                         + list(Path(d).rglob("*.onnx"))
                         + list(Path(d).rglob("*.safetensors")))
                if len(files) > 0:
                    ok(f"找到模型权重: {d} ({len(files)}个文件)")
                    found = True
                    break

        if not found:
            warn("未找到MinerU模型权重，需要下载（约2GB）")
            self.results["mineru_models"] = False
        else:
            self.results["mineru_models"] = True

    # ---------- 5. API连通性 ----------
    def check_api(self):
        header("5/6 检查智谱API连通性")

        if not ZHIPU_API_KEY:
            warn("未填写API Key，跳过API测试")
            warn("请在脚本顶部 ZHIPU_API_KEY 处填入你的Key")
            self.results["api"] = None  # None表示未测试
            return

        try:
            req = urllib.request.Request(
                ZHIPU_TEST_URL,
                headers={"Authorization": f"Bearer {ZHIPU_API_KEY}"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                models = [m.get("id", "") for m in data.get("data", [])]

                if "glm-5.3-flash" in models or any("glm-5" in m for m in models):
                    ok(f"API连通成功，glm-5.3-flash可用")
                    self.results["api"] = True
                else:
                    warn(f"API连通但未见glm-5.3-flash模型")
                    info(f"可用模型: {', '.join(models[:10])}")
                    self.results["api"] = True  # 连通了就算通过
        except urllib.error.HTTPError as e:
            if e.code == 401:
                fail("API Key无效！请检查")
            else:
                fail(f"API请求失败: HTTP {e.code}")
            self.results["api"] = False
        except Exception as e:
            fail(f"API测试异常: {e}")
            self.results["api"] = False

    # ---------- 6. 磁盘空间 ----------
    def check_disk(self):
        header("6/6 检查磁盘空间")

        # 检查输出目录所在盘符
        output_root = os.environ.get("REHAB_OUTPUT_DIR", r"D:\rehab_books\output")
        drive = os.path.splitdrive(output_root)[0] or os.path.splitdrive(os.getcwd())[0]
        # 若配置的输出盘符不存在（如本机只有C盘），回退到当前系统盘符
        if not os.path.exists(drive + "\\"):
            drive = os.path.splitdrive(os.getcwd())[0] or "C:"
            warn(f"配置的输出盘符不存在，回退检查系统盘 {drive}:")

        try:
            usage = shutil.disk_usage(drive + "\\")
            free_gb = usage.free / (1024**3)
            total_gb = usage.total / (1024**3)

            info(f"磁盘 {drive}: 剩余 {free_gb:.1f}GB / 总共 {total_gb:.1f}GB")

            if free_gb > 10:
                ok("磁盘空间充足 (>10GB)")
                self.results["disk"] = True
            elif free_gb > 3:
                warn(f"磁盘空间偏紧 ({free_gb:.1f}GB)，建议预留>10GB")
                self.results["disk"] = True
            else:
                fail(f"磁盘空间不足 ({free_gb:.1f}GB)！MinerU模型+中间产物需约5GB")
                self.results["disk"] = False
        except Exception as e:
            warn(f"无法检查磁盘空间: {e}")
            self.results["disk"] = None

    # ============================================================
    # 【自动修复】
    # ============================================================

    def auto_fix(self):
        """尝试自动修复缺失的依赖"""
        header("🔧 自动修复")

        fixed_any = False

        # 修复1: 创建Conda环境 + 安装MinerU
        if self.results.get("conda") and not self.results.get("mineru"):
            info(f"正在创建Conda环境 '{CONDA_ENV_NAME}' 并安装MinerU...")
            info("这可能需要5~15分钟，请耐心等待...")

            # 创建环境
            cmd_create = [
                self.conda_path, "create", "-n", CONDA_ENV_NAME,
                f"python={CONDA_PYTHON_VERSION}", "-y"
            ]
            ret = subprocess.run(cmd_create, capture_output=True, text=True, timeout=600)
            if ret.returncode != 0:
                fail(f"创建Conda环境失败: {ret.stderr[:300]}")
            else:
                ok(f"Conda环境 '{CONDA_ENV_NAME}' 创建成功")

                # 安装MinerU
                info("正在安装MinerU...")
                cmd_install = [
                    self.conda_path, "run", "-n", CONDA_ENV_NAME,
                    "pip", "install", "-U", "mineru[all]",
                    "-i", "https://mirrors.aliyun.com/pypi/simple"
                ]
                ret = subprocess.run(cmd_install, capture_output=True, text=True, timeout=1800)
                if ret.returncode != 0:
                    fail(f"MinerU安装失败: {ret.stderr[:500]}")
                else:
                    ok("MinerU安装成功！")
                    self.results["mineru"] = True
                    fixed_any = True

        # 修复2: 下载MinerU模型
        if self.results.get("mineru") and not self.results.get("mineru_models"):
            info("正在下载MinerU模型权重（约2GB）...")
            info("这可能需要3~10分钟，取决于网速...")

            cmd_download = [
                self.conda_path, "run", "-n", CONDA_ENV_NAME,
                "mineru-models-download"
            ]
            ret = subprocess.run(cmd_download, capture_output=True, text=True, timeout=3600)
            if ret.returncode != 0:
                fail(f"模型下载失败: {ret.stderr[:500]}")
                warn("你也可以手动下载: conda activate mineru && mineru-models-download")
            else:
                ok("MinerU模型权重下载完成！")
                self.results["mineru_models"] = True
                fixed_any = True

        if not fixed_any:
            info("没有需要自动修复的项目")

    # ============================================================
    # 【最终报告】
    # ============================================================

    def print_report(self):
        header("📋 环境诊断报告")

        items = [
            ("Python ≥3.9",      self.results.get("python")),
            ("Conda",            self.results.get("conda")),
            (f"Conda环境'{CONDA_ENV_NAME}'", self.results.get("conda_env")),
            ("MinerU",           self.results.get("mineru")),
            ("MinerU模型权重",   self.results.get("mineru_models")),
            ("智谱API",          self.results.get("api")),
            ("磁盘空间",         self.results.get("disk")),
        ]

        all_pass = True
        for name, status in items:
            if status is True:
                ok(name)
            elif status is False:
                fail(name)
                all_pass = False
            else:
                warn(f"{name} (未检测)")

        print()
        if all_pass:
            print(f"  {GREEN}{BOLD}🎉 所有环境检查通过！可以运行主脚本了。{RESET}")
            print(f"  {CYAN}运行命令: python pipeline.py{RESET}")
        else:
            print(f"  {RED}{BOLD}⚠️ 部分检查未通过，请根据上方提示修复。{RESET}")

        # 如果mineru在conda环境中，提示主脚本如何调用
        if self.mineru_path:
            print(f"\n  {CYAN}💡 提示: 主脚本中MinerU CLI路径为:{RESET}")
            print(f"     {self.mineru_path}")
            print(f"  {CYAN}   如主脚本找不到mineru命令，请将此路径填入配置区{RESET}")

        print()
        return all_pass

    # ============================================================
    # 【主流程】
    # ============================================================

    def run(self):
        print(f"\n{BOLD}╔══════════════════════════════════════════════════╗")
        print(f"║   康复医学LoRA流水线 - 环境依赖检查工具          ║")
        print(f"║   运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}              ║")
        print(f"╚══════════════════════════════════════════════════╝{RESET}")

        # 第一轮检查
        self.check_python()
        self.check_conda()
        self.check_mineru()
        self.check_mineru_models()
        self.check_api()
        self.check_disk()

        # 自动修复
        needs_fix = (
            (self.results.get("conda") and not self.results.get("mineru")) or
            (self.results.get("mineru") and not self.results.get("mineru_models"))
        )
        if needs_fix:
            self.auto_fix()
            # 修复后重新检查相关项
            if not self.results.get("mineru"):
                self.check_mineru()
            if not self.results.get("mineru_models"):
                self.check_mineru_models()

        # 输出报告
        return self.print_report()


# ============================================================
# 【入口】
# ============================================================

if __name__ == "__main__":
    checker = EnvChecker()
    success = checker.run()
    sys.exit(0 if success else 1)