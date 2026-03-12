"""
Translator - 使用 Grok (XAI) API 翻译文本为中文
用于将 ArXiv 论文摘要翻译成简体中文，以及生成博客文章摘要
已从 Gemini 迁移至 Grok API（OpenAI 兼容格式）
"""
import sys
import time
import logging
import httpx

logger = logging.getLogger(__name__)

# Force UTF-8 stdout for Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Import from centralized config
try:
    from config import (
        XAI_API_KEY, XAI_BASE_URL, XAI_TRANSLATE_MODEL,
        GROK_TIMEOUT, XAI_TRANSLATE_MAX_RETRIES
    )
except ImportError:
    from src.config import (
        XAI_API_KEY, XAI_BASE_URL, XAI_TRANSLATE_MODEL,
        GROK_TIMEOUT, XAI_TRANSLATE_MAX_RETRIES
    )

# 保留向后兼容：report_generator.py 通过 GEMINI_AVAILABLE 判断是否可用
GEMINI_API_KEY = XAI_API_KEY  # 兼容旧的检查逻辑


def _call_grok(prompt: str, max_tokens: int = 1024, temperature: float = 0.3) -> str:
    """
    调用 Grok API（OpenAI 兼容格式）。

    Args:
        prompt: 提示词
        max_tokens: 最大输出 token 数
        temperature: 温度参数

    Returns:
        API 返回的文本，失败返回空字符串
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {XAI_API_KEY}"
    }

    payload = {
        "model": XAI_TRANSLATE_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    for attempt in range(XAI_TRANSLATE_MAX_RETRIES):
        try:
            response = httpx.post(
                XAI_BASE_URL,
                headers=headers,
                json=payload,
                timeout=GROK_TIMEOUT
            )
            response.raise_for_status()

            data = response.json()
            result = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            if result:
                return result.strip()
            else:
                if attempt < XAI_TRANSLATE_MAX_RETRIES - 1:
                    logger.warning(f"Grok 返回空结果，重试 ({attempt + 1}/{XAI_TRANSLATE_MAX_RETRIES})...")
                    time.sleep(2 ** attempt)
                    continue
                return ""

        except (httpx.HTTPError, httpx.TimeoutException, ValueError, KeyError) as e:
            if attempt < XAI_TRANSLATE_MAX_RETRIES - 1:
                logger.warning(f"Grok API 调用失败 ({attempt + 1}/{XAI_TRANSLATE_MAX_RETRIES}): {e}")
                time.sleep(2 ** attempt)
                continue
            logger.error(f"Grok API 最终失败: {e}")
            return ""

    return ""


def translate_to_chinese(text: str, max_chars: int = 100) -> str:
    """
    将英文文本翻译成简体中文。

    Args:
        text: 要翻译的英文文本
        max_chars: 输出的最大字符数（用于 brief）

    Returns:
        翻译后的中文文本，如果失败则返回原文
    """
    if not XAI_API_KEY:
        logger.warning("XAI_API_KEY 未配置，跳过翻译")
        return text[:max_chars] + "..." if len(text) > max_chars else text

    if not text or len(text) < 10:
        return text

    prompt = f"""请将以下学术论文摘要完整翻译成简体中文，要求：
1. 保持学术风格，用词精准
2. 完整翻译全部内容，不要省略任何信息
3. 只输出翻译结果，不要添加任何解释

原文：
{text}"""

    result = _call_grok(prompt, max_tokens=1024, temperature=0.3)
    if result:
        return result
    return text[:max_chars] + "..." if len(text) > max_chars else text


def translate_summary_pair(summary: str) -> tuple[str, str]:
    """
    为 ArXiv 论文生成两层摘要（中文）。

    Args:
        summary: 英文原始摘要

    Returns:
        (brief_cn, detail_cn) - 短摘要和详细摘要的中文版本
    """
    if not summary:
        return ("", "")

    # Brief: 翻译前100字
    brief_cn = translate_to_chinese(summary[:200], max_chars=80)

    # Detail: 翻译完整摘要
    detail_cn = translate_to_chinese(summary, max_chars=500)

    return (brief_cn, detail_cn)


def summarize_blog_article(content: str, mode: str = "brief") -> str:
    """
    为技术博客文章生成情报简报风格的中文摘要。

    Args:
        content: 博客文章的完整内容（Markdown格式）
        mode: "brief" (一句话摘要) 或 "detail" (深度分析)

    Returns:
        中文摘要
    """
    if not XAI_API_KEY or not content or len(content) < 50:
        return ""

    if mode == "brief":
        prompt = f"""请阅读以下技术博客文章，用一句话中文概括核心观点（最多100字）。
要求：
- 直接说重点，不要"本文介绍了..."这种开头
- 忽略作者信息、日期、URL等元数据
- 突出技术洞察或实用价值

文章内容：
{content[:2000]}"""
        max_tokens = 256
    else:  # detail
        prompt = f"""请作为技术情报分析师，阅读以下博客文章并生成中文深度分析报告。

要求：
1. 忽略作者信息、URL、图片链接等元数据
2. 提取核心技术观点和实践经验
3. 用3-4个段落组织：背景、关键发现、技术细节、实用价值
4. 语言风格：专业但易懂，适合技术人士快速阅读
5. 总长度控制在300-500字

文章内容：
{content[:6000]}"""
        max_tokens = 1024

    result = _call_grok(prompt, max_tokens=max_tokens, temperature=0.4)
    return result


if __name__ == "__main__":
    # Test translation
    test_text = "Adapting large pretrained models to new tasks efficiently and continually is crucial for real-world deployment but remains challenging due to catastrophic forgetting."
    print("原文:", test_text)
    print("翻译:", translate_to_chinese(test_text, 80))
