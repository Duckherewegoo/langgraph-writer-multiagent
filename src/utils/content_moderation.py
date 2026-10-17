from pathlib import Path
from typing import Tuple, Set, Optional
import logging

logger = logging.getLogger(__name__)

#优雅降级。没有 ahocorasick 也能跑，只是慢一点
try:
    import ahocorasick
    HAS_AC = True
except ImportError:
    HAS_AC = False
    logger.warning("未安装 pyahocorasick，敏感词匹配将降级为纯 Python 模式（性能略低）。")


# 全局缓存，惰性加载
_SENSITIVE_AC = None
_FALLBACK_WORDS: Set[str] = set()


def _load_vocab(vocab_dir: str) -> Set[str]:
    """加载词库，返回去重后的词条集合"""
    vocab_path = Path(vocab_dir)
    if not vocab_path.exists() or not vocab_path.is_dir():
        logger.warning(f"敏感词库目录不存在: {vocab_dir}，审核功能将放行所有内容。")
        return set()

    all_words = set()
    for txt_file in vocab_path.glob("*.txt"):
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.strip()
                    if word:
                        all_words.add(word)
        except UnicodeDecodeError:
            logger.warning(f"文件 {txt_file.name} 编码非 utf-8，已跳过")
        except Exception as e:
            logger.error(f"读取词库文件 {txt_file.name} 失败: {e}")
            
    return all_words


def _get_automaton(vocab_dir: str):
    """第一次调用时再初始化，避免 import 时读盘"""
    global _SENSITIVE_AC, _FALLBACK_WORDS
    
    # 如果已经加载过了，直接返回缓存
    if _SENSITIVE_AC is not None or _FALLBACK_WORDS:
        return

    all_words = _load_vocab(vocab_dir)
    if not all_words:
        return

    if HAS_AC:
        automaton = ahocorasick.Automaton()
        for word in all_words:
            automaton.add_word(word, word)
        automaton.make_automaton()
        _SENSITIVE_AC = automaton
        logger.info(f"词库加载完成（AC自动机）：共加载 {len(all_words)} 个敏感词")
    else:
        _FALLBACK_WORDS = all_words
        logger.info(f"词库加载完成（纯Python）：共加载 {len(all_words)} 个敏感词")


def moderate_content(content: str, vocab_dir: Optional[str] = None) -> Tuple[bool, str]:
    """
    内容安全审核，找到第一个匹配就返回。
    :param vocab_dir: 词库路径。建议在 main.py 启动时传绝对路径进来。
    """
    # 确保初始化（如果外部没传，用默认相对路径兜底，或直接跳过）
    if vocab_dir:
        _get_automaton(vocab_dir)

    # 如果初始化失败（没词库/目录不对），放行通过（安全兜底，避免因审核组件挂掉导致业务瘫痪）
    if _SENSITIVE_AC is None and not _FALLBACK_WORDS:
        return True, ""

    # 命中即返回
    if _SENSITIVE_AC is not None:
        for _, word in _SENSITIVE_AC.iter(content):
            return False, f"包含敏感内容：{word}"
    else:
        # 降级方案：简单子串匹配
        for word in _FALLBACK_WORDS:
            if word in content:
                return False, f"包含敏感内容：{word}"

    return True, ""
