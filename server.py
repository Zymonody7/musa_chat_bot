"""FastAPI 服务"""
import os
import time
import logging
from typing import Optional, List, Dict
from contextlib import asynccontextmanager

# MUSA 设备需要 eager attention（必须在 import torch 之前设置）
os.environ["TRANSFORMERS_ATTN_IMPLEMENTATION"] = "eager"
os.environ["PYTORCH_MUSA_ALLOC_CONF"] = "max_split_size_mb:128"

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
import torch_musa  # noqa: F401
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextIteratorStreamer
from threading import Thread
import json
from datetime import datetime
from collections import defaultdict

from rag.index import RAGIndex
from rag.prompts import build_prompt

# 配置日志m
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局变量
rag_index: Optional[RAGIndex] = None
llm_tokenizer: Optional[AutoTokenizer] = None
llm_model: Optional[AutoModelForCausalLM] = None
device: str = "musa"  # 设备类型
session_history: Dict[str, List[str]] = defaultdict(list)  # 每个session的最近3次问题的关键词
sessions: Dict[str, List[Dict]] = defaultdict(list)  # 每个session的对话记录


class AskRequest(BaseModel):
    question: str
    session_id: Optional[str] = None  # 会话ID，如果为空则创建新会话


class AskResponse(BaseModel):
    answer: str
    sources: List[Dict]
    latency_ms: float
    retrieve_ms: float
    generate_ms: float
    session_id: str  # 返回会话ID


class SessionMessage(BaseModel):
    role: str  # 'user' or 'model'
    content: str
    timestamp: float
    sources: Optional[List[Dict]] = None


class Session(BaseModel):
    session_id: str
    title: str
    created_at: float
    updated_at: float
    messages: List[SessionMessage]


class SessionListResponse(BaseModel):
    sessions: List[Dict]  # 简化的session信息列表


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载模型，关闭时清理"""
    global rag_index, llm_tokenizer, llm_model, device
    
    # 从环境变量获取配置
    embedding_model_path = os.getenv("EMBEDDING_MODEL_PATH", "./models/embedding_model")
    llm_model_path = os.getenv("LLM_MODEL_PATH", "/root/autodl-tmp/local-model")
    index_dir = os.getenv("INDEX_DIR", "./rag_store")
    
    # 使用 musa GPU
    device = "musa"
    logger.info(f"使用设备: {device} (musa GPU)")
    
    # 加载 RAG 索引
    logger.info("加载 RAG 索引...")
    try:
        rag_index = RAGIndex(
            embedding_model_path=embedding_model_path,
            index_dir=index_dir
        )
        rag_index.load_index()
        logger.info("RAG 索引加载完成")
    except Exception as e:
        logger.error(f"加载 RAG 索引失败: {e}")
        raise
    
    # 加载 LLM 模型
    logger.info(f"加载 LLM 模型: {llm_model_path}...")
    logger.info(f"使用 musa GPU 加速")
    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(
            llm_model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        # 使用 musa GPU 加载配置（优化：不使用 device_map，手动移动）
        load_kwargs = {
            "trust_remote_code": True,
            "local_files_only": True,
            "torch_dtype": torch.float16,
            "attn_implementation": "eager",  # musa 设备必需
            "device_map": None,  # 不使用 device_map，手动移动更快
        }
        
        llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model_path,
            **load_kwargs
        )
        
        # 手动移动到 musa 设备
        if hasattr(torch, 'musa') and torch.musa.is_available():
            torch.musa.set_device(0)
            llm_model = llm_model.to("musa", non_blocking=True)  # 使用 non_blocking
            logger.info("模型已移动到 musa 设备")
        
        llm_model.eval()
        logger.info("LLM 模型加载完成")
        
        # 可选：torch.compile 加速推理（参考 serve.py）
        try:
            if hasattr(torch, "compile") and device != "cpu":
                logger.info("正在编译模型以加速推理...")
                llm_model = torch.compile(llm_model, mode="reduce-overhead", fullgraph=False)
                logger.info("✓ 模型编译完成")
        except Exception as e:
            logger.warning(f"torch.compile 失败，继续使用未编译模型: {e}")
        
        # 预热模型（参考 serve.py）
        logger.info("正在预热模型...")
        try:
            with torch.inference_mode():  # 使用 inference_mode 而不是 no_grad
                warmup_prompt = "<|im_start|>user\n测试<|im_end|>\n<|im_start|>assistant\n"
                warmup_inputs = llm_tokenizer(
                    warmup_prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=64,
                )
                warmup_inputs = {k: v.to(llm_model.device, non_blocking=True) for k, v in warmup_inputs.items()}
                _ = llm_model.generate(
                    **warmup_inputs,
                    max_new_tokens=8,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=llm_tokenizer.eos_token_id,
                    eos_token_id=llm_tokenizer.eos_token_id,
                    num_beams=1,
                )
            logger.info("✓ 预热完成")
        except Exception as e:
            logger.warning(f"预热失败（可忽略）: {e}")
    except Exception as e:
        logger.error(f"加载 LLM 模型失败: {e}")
        raise
    
    yield
    
    # 清理
    logger.info("关闭服务，清理资源...")
    rag_index = None
    llm_tokenizer = None
    llm_model = None


app = FastAPI(
    title="离线 PDF RAG 推理系统",
    description="基于本地 PDF 教材和本地大模型的问答系统",
    lifespan=lifespan
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境建议指定具体域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法，包括 OPTIONS
    allow_headers=["*"],  # 允许所有请求头
)


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok"}


@app.get("/pdf")
async def get_pdf():
    """获取PDF文件"""
    pdf_path = os.getenv("PDF_PATH", "./rag/statics/PMPP-3rd-Edition.pdf")
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF文件不存在")
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=os.path.basename(pdf_path),
        headers={
            "Content-Disposition": f'inline; filename="{os.path.basename(pdf_path)}"',
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Expose-Headers": "*",
            "Cross-Origin-Resource-Policy": "cross-origin",
            "Cross-Origin-Embedder-Policy": "unsafe-none",
        }
    )


@app.options("/pdf")
async def options_pdf():
    """处理PDF文件的OPTIONS请求（CORS预检）"""
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, OPTIONS",
        "Access-Control-Allow-Headers": "*",
    }


@app.get("/page/{page_num}")
async def get_page(page_num: int):
    """获取指定页面的内容"""
    global rag_index
    
    if rag_index is None:
        raise HTTPException(status_code=500, detail="索引未加载")
    
    try:
        # 从 chunks 中查找该页面的所有内容
        page_chunks = []
        for chunk in rag_index.chunks:
            if chunk["meta"].get("pdf_page") == page_num:
                page_chunks.append({
                    "text": chunk["text"],
                    "meta": chunk["meta"],
                    "chunk_idx": chunk["meta"].get("chunk_idx", 0),
                })
        
        if not page_chunks:
            raise HTTPException(status_code=404, detail=f"未找到第 {page_num} 页的内容")
        
        # 按 chunk_idx 排序
        page_chunks.sort(key=lambda x: x["chunk_idx"])
        
        # 合并文本
        full_text = "\n\n".join([chunk["text"] for chunk in page_chunks])
        
        # 获取章节信息
        chapter = page_chunks[0]["meta"].get("chapter") if page_chunks else None
        section = page_chunks[0]["meta"].get("section") if page_chunks else None
        
        return {
            "page": page_num,
            "text": full_text,
            "chapter": chapter,
            "section": section,
            "chunks": page_chunks
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取页面内容时出错: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"内部错误: {str(e)}")


def _expand_query(question: str, session_id: str) -> str:
    """使用会话历史扩展查询（简单实现）"""
    global session_history
    
    history = session_history.get(session_id, [])
    if not history:
        return question
    
    # 简单拼接：问题 + 最近的关键词
    expanded = question + " " + " ".join(history[-3:])
    return expanded


def _update_session_history(question: str, session_id: str):
    """更新会话历史（提取关键词，这里简单用原问题）"""
    global session_history
    # 简单实现：保存问题的前几个词作为关键词
    words = question.split()[:5]  # 取前 5 个词
    session_history[session_id].extend(words)
    # 保持最近 3 次问题的关键词（约 15 个词）
    if len(session_history[session_id]) > 15:
        session_history[session_id] = session_history[session_id][-15:]


def _save_message_to_session(session_id: str, role: str, content: str, sources: Optional[List[Dict]] = None):
    """保存消息到session"""
    global sessions
    message = {
        "role": role,
        "content": content,
        "timestamp": time.time(),
        "sources": sources or []
    }
    sessions[session_id].append(message)
    # 限制每个session最多保存1000条消息
    if len(sessions[session_id]) > 1000:
        sessions[session_id] = sessions[session_id][-1000:]


def _generate_session_id() -> str:
    """生成新的session ID"""
    return f"session_{int(time.time() * 1000)}"


def _generate_answer_stream(question: str, sources: List[Dict], session_id: str):
    """生成答案的流式生成器"""
    global llm_tokenizer, llm_model, device
    
    # 保存用户消息到session
    _save_message_to_session(session_id, "user", question)
    
    # 先发送检索信息
    yield f"data: {json.dumps({'type': 'retrieve', 'sources': sources, 'session_id': session_id}, ensure_ascii=False)}\n\n"
    
    # 构建提示词
    prompt_data = build_prompt(question, sources)
    messages = prompt_data["messages"]
    
    # 应用 chat template
    prompt_text = llm_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize（优化：使用 non_blocking）
    model_device = next(llm_model.parameters()).device
    inputs = llm_tokenizer(
        prompt_text,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    # 使用 non_blocking=True 加速数据传输
    inputs = {k: v.to(model_device, non_blocking=True) for k, v in inputs.items()}
    
    # 创建流式迭代器（不跳过特殊token，保留think内容）
    streamer = TextIteratorStreamer(
        llm_tokenizer,
        skip_prompt=True,
        skip_special_tokens=False,  # 保留think标记
        timeout=300.0
    )
    
    # 定义生成函数（在后台线程中运行，使用 inference_mode）
    def generate_with_inference_mode():
        with torch.inference_mode():  # 使用 inference_mode 比 no_grad 更快
            llm_model.generate(
                **inputs,
                max_new_tokens=2048,  # 增加生成长度上限，让模型完整回答
                do_sample=False,
                use_cache=True,
                pad_token_id=llm_tokenizer.eos_token_id,
                eos_token_id=llm_tokenizer.eos_token_id,
                repetition_penalty=1.1,
                num_beams=1,
                early_stopping=True,  # 遇到 EOS 自动停止
                streamer=streamer,
            )
    
    thread = Thread(target=generate_with_inference_mode)
    thread.start()
    
    # 流式输出token
    generated_text = ""
    try:
        for new_text in streamer:
            generated_text += new_text
            yield f"data: {json.dumps({'type': 'token', 'token': new_text, 'text': generated_text}, ensure_ascii=False)}\n\n"
        
        # 保存模型回答到session
        _save_message_to_session(session_id, "model", generated_text, sources)
        
        # 发送完成信号
        yield f"data: {json.dumps({'type': 'done', 'full_text': generated_text, 'session_id': session_id}, ensure_ascii=False)}\n\n"
    except Exception as e:
        logger.error(f"流式生成错误: {e}")
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)}, ensure_ascii=False)}\n\n"
    finally:
        thread.join(timeout=5)
        # 清理缓存
        if device == "musa" and hasattr(torch.musa, 'empty_cache'):
            torch.musa.empty_cache()


@app.post("/ask/stream")
async def ask_stream(request: AskRequest):
    """流式问答接口"""
    global rag_index, llm_tokenizer, llm_model
    
    if rag_index is None or llm_tokenizer is None or llm_model is None:
        raise HTTPException(status_code=500, detail="模型未加载")
    
    try:
        # 生成或使用session_id
        session_id = request.session_id or _generate_session_id()
        
        # 检索阶段
        logger.info(f"开始检索: {request.question}, session_id: {session_id}")
        
        # 扩展查询（使用会话历史）
        expanded_query = _expand_query(request.question, session_id)
        
        # 搜索相关 chunks
        search_results = rag_index.search(expanded_query, top_k=8)
        
        # 按页面聚合
        sources = rag_index.aggregate_by_page(search_results, max_sources=4)
        
        logger.info(f"检索完成，找到 {len(sources)} 个sources")
        
        if not sources:
            raise HTTPException(
                status_code=404,
                detail="未找到相关的教材内容"
            )
        
        # 更新会话历史
        _update_session_history(request.question, session_id)
        
        # 返回流式响应
        return StreamingResponse(
            _generate_answer_stream(request.question, sources, session_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理请求时出错: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"内部错误: {str(e)}")


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """问答接口（非流式，保持兼容性）"""
    global rag_index, llm_tokenizer, llm_model
    
    if rag_index is None or llm_tokenizer is None or llm_model is None:
        raise HTTPException(status_code=500, detail="模型未加载")
    
    start_time = time.time()
    
    try:
        # 生成或使用session_id
        session_id = request.session_id or _generate_session_id()
        
        # 1. 检索阶段
        retrieve_start = time.time()
        logger.info(f"开始检索: {request.question}, session_id: {session_id}")
        
        # 扩展查询（使用会话历史）
        expanded_query = _expand_query(request.question, session_id)
        
        # 搜索相关 chunks
        search_results = rag_index.search(expanded_query, top_k=8)
        
        # 按页面聚合
        sources = rag_index.aggregate_by_page(search_results, max_sources=4)
        
        retrieve_ms = (time.time() - retrieve_start) * 1000
        logger.info(f"检索完成，找到 {len(sources)} 个sources，耗时: {retrieve_ms:.2f}ms")
        
        if not sources:
            raise HTTPException(
                status_code=404,
                detail="未找到相关的教材内容"
            )
        
        # 2. 生成阶段
        generate_start = time.time()
        
        # 构建提示词
        prompt_data = build_prompt(request.question, sources)
        messages = prompt_data["messages"]
        
        # 应用 chat template
        prompt_text = llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize（优化：使用 non_blocking 和 inference_mode）
        model_device = next(llm_model.parameters()).device
        inputs = llm_tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        # 使用 non_blocking=True 加速数据传输
        inputs = {k: v.to(model_device, non_blocking=True) for k, v in inputs.items()}
        
        # 生成（使用 musa GPU 加速，优化：使用 inference_mode）
        logger.info(f"开始生成，输入长度: {inputs['input_ids'].shape[1]}")
        generate_start_inner = time.time()
        
        with torch.inference_mode():  # 使用 inference_mode 比 no_grad 更快
            outputs = llm_model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=2048,  # 增加生成长度上限，让模型完整回答
                use_cache=True,
                pad_token_id=llm_tokenizer.eos_token_id,
                eos_token_id=llm_tokenizer.eos_token_id,
                repetition_penalty=1.1,  # 防止重复
                num_beams=1,  # 不使用beam search以加快速度
                early_stopping=True,  # 遇到EOS立即停止，让模型自然结束
            )
        
        logger.info(f"生成完成，输出长度: {outputs.shape[1]}, 耗时: {(time.time() - generate_start_inner):.2f}秒")
        
        # 解码（只取新生成的 token）
        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_length:].cpu()  # 移到 CPU 再解码
        answer = llm_tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        # 释放中间张量
        del outputs, generated_ids
        if device == "musa" and hasattr(torch.musa, 'empty_cache'):
            torch.musa.empty_cache()
        
        generate_ms = (time.time() - generate_start) * 1000
        
        # 更新会话历史
        _update_session_history(request.question, session_id)
        
        # 保存消息到session
        _save_message_to_session(session_id, "user", request.question)
        _save_message_to_session(session_id, "model", answer, sources)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # 构造响应
        return AskResponse(
            answer=answer,
            sources=sources,
            latency_ms=round(latency_ms, 2),
            retrieve_ms=round(retrieve_ms, 2),
            generate_ms=round(generate_ms, 2),
            session_id=session_id
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理请求时出错: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"内部错误: {str(e)}")


@app.get("/sessions")
async def get_sessions():
    """获取所有session列表"""
    global sessions
    session_list = []
    for session_id, messages in sessions.items():
        if messages:
            # 使用第一条用户消息作为标题
            first_user_msg = next((msg for msg in messages if msg["role"] == "user"), None)
            title = first_user_msg["content"][:50] if first_user_msg else "新对话"
            if len(title) > 50:
                title = title[:47] + "..."
            
            session_list.append({
                "session_id": session_id,
                "title": title,
                "created_at": messages[0]["timestamp"] if messages else time.time(),
                "updated_at": messages[-1]["timestamp"] if messages else time.time(),
                "message_count": len(messages)
            })
    
    # 按更新时间倒序排列
    session_list.sort(key=lambda x: x["updated_at"], reverse=True)
    return {"sessions": session_list}


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """获取指定session的详细信息"""
    global sessions
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session不存在")
    
    messages = sessions[session_id]
    if not messages:
        raise HTTPException(status_code=404, detail="Session为空")
    
    first_user_msg = next((msg for msg in messages if msg["role"] == "user"), None)
    title = first_user_msg["content"][:50] if first_user_msg else "新对话"
    if len(title) > 50:
        title = title[:47] + "..."
    
    return {
        "session_id": session_id,
        "title": title,
        "created_at": messages[0]["timestamp"],
        "updated_at": messages[-1]["timestamp"],
        "messages": messages
    }


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除指定session"""
    global sessions, session_history
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session不存在")
    
    del sessions[session_id]
    if session_id in session_history:
        del session_history[session_id]
    
    return {"message": "Session已删除", "session_id": session_id}


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    uvicorn.run(app, host=host, port=port)

