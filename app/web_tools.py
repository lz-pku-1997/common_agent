"""真实联网搜索：对外一个工具，搜索与网页提取由千问 Responses API 编排。"""

import json

import httpx
from langchain_core.tools import tool

from app.config import require_environment_variable
from app.tool_errors import FixableError, NonRetryableError, RetryableError


@tool
async def web_search(query: str) -> str:
    """联网查找当前信息，返回回答、网页提取内容与来源 URL；查询会发到外部服务。"""
    if not query.strip():
        raise FixableError("搜索词不能为空")  # 空查询是模型能通过改参数修好的错误。
    settings = {  # 联网搜索固定使用 strong 模型，复用配置，不随会话档位切换。
        "api_key": require_environment_variable("LLM_API_KEY"),
        "base_url": require_environment_variable("LLM_BASE_URL"),
        "model": require_environment_variable("LLM_STRONG_MODEL"),
    }
    try:
        async with httpx.AsyncClient(timeout=60) as client:  # HTTP 超时只约束本次请求，不是通用工具取消机制。
            response = await client.post(
                str(settings["base_url"]).rstrip("/") + "/responses",
                headers={"Authorization": f"Bearer {settings['api_key']}"},
                json={
                    "model": settings["model"],
                    "reasoning": {"effort": "low"},  # 搜索工具负责查事实，不沿用供应商默认 xhigh 的长推理。
                    "input": f"请联网查证以下问题，给出结论和来源；没有查到就明确说明：{query}",
                    "tools": [{"type": "web_search"}, {"type": "web_extractor"}],  # 提取不能单独用；需要降级时只删第二项。
                },
            )
    except httpx.TransportError as error:
        raise RetryableError("联网服务连接失败或超时，可以原样补试") from error  # 只读搜索可安全重复，但补试仍可能产生费用。
    if response.status_code in {408, 429} or response.status_code >= 500:
        raise RetryableError(f"联网服务暂时不可用（HTTP {response.status_code}）")
    if response.status_code == 400:
        raise FixableError("联网请求被拒绝；请缩短或修改查询，仍失败请检查服务是否支持内置搜索")
    if response.is_error:
        raise NonRetryableError(f"联网服务拒绝请求（HTTP {response.status_code}），请检查端点、权限和额度")  # 不返回请求头或响应原文，避免泄露凭据。
    data = response.json()
    if data.get("error") or data.get("status") in {"failed", "incomplete", "cancelled"}:
        raise NonRetryableError("联网服务没有完成搜索，不能把部分结果当作成功")
    if not any(item.get("type") == "web_search_call" for item in data.get("output", [])):
        raise NonRetryableError("服务没有实际执行联网搜索，不能把模型记忆中的回答当作联网结果")  # 当前套餐不接受 required，用真实调用记录核验。
    blocks = []
    for item in data.get("output", []):  # 跳过 reasoning，只保留模型正文和真正的搜索证据。
        kind = item.get("type")
        if kind == "message":
            blocks.extend(part["text"] for part in item.get("content", []) if part.get("type") == "output_text")
        elif kind == "web_search_call":
            blocks.append("搜索来源：" + json.dumps(item.get("action", {}), ensure_ascii=False))
        elif kind == "web_extractor_call":
            blocks.append("网页提取：" + json.dumps({key: item.get(key) for key in ("urls", "goal", "output")}, ensure_ascii=False))
    if not blocks:
        raise NonRetryableError("联网服务未返回可阅读的结果")
    return "\n\n".join(blocks)  # 主循环按 Registry 的 web 来源统一隔离，并处理超长结果。


WEB_TOOLS = [web_search]
