"""提示词模板模块"""
from typing import List, Dict


def format_sources(sources: List[Dict]) -> str:
    """
    格式化 sources 为字符串
    
    Args:
        sources: sources 列表，每个包含 pdf_page, chapter, section, snippet
    
    Returns:
        格式化的 sources 文本
    """
    formatted = []
    for src in sources:
        parts = [f"P{src['pdf_page']}"]
        if src.get("chapter"):
            parts.append(src["chapter"])
        if src.get("section"):
            parts.append(src["section"])
        
        header = f"[Source {' | '.join(parts)}]"
        formatted.append(f"{header}\n{src['snippet']}")
    
    return "\n\n".join(formatted)


def build_prompt(question: str, sources: List[Dict]) -> Dict[str, str]:
    """
    构建 Qwen3 提示词（使用 apply_chat_template）
    
    Args:
        question: 问题
        sources: sources 列表
    
    Returns:
        Dict: 包含 messages 列表，可直接用于 tokenizer.apply_chat_template
    """
    system_prompt = """你是一个基于教材内容回答问题的助手。

要求：
1. 必须基于提供的 Sources 回答问题
2. 每个关键结论后必须标注引用 [Pxxx]（xxx 是 PDF 页码）
3. 如果 Sources 不足以回答，明确说"教材证据不足"，并建议查阅的章节/关键词（可根据 sources 的 chapter/section）"""

    sources_text = format_sources(sources)
    
    user_content = f"""问题：{question}

Sources：
{sources_text}"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    return {"messages": messages}


