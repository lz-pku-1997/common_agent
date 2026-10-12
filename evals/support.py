"""评测共用的文本与统计辅助函数。"""


# 从字符串或文本内容块中提取纯文本，供回答展示和裁判结果解析使用。
# 字符串示例："  你好  " -> "你好"，strip() 去掉首尾空白。
# 内容块示例：[{"type": "text", "text": "第一段"}, {"type": "image", "url": "..."},
#              {"type": "text", "text": "第二段"}] -> "第一段\n第二段"。
# 列表只提取字典中 type="text" 的 text；跳过图片等其他块，用换行拼接文字。
# 其他输入示例：123 -> "123"，用 str() 转成字符串。
def message_text(content) -> str:
    if isinstance(content, str):
        return content.strip()  # 已经是文字，只去掉首尾空白。
    if isinstance(content, list):
        return "\n".join(
            str(block.get("text", "")) for block in content  # 遍历每块，取出文字；缺少 text 时取空串。
            if isinstance(block, dict) and block.get("type") == "text"  # 只保留文本块。
        ).strip()
    return str(content).strip()  # 其他类型转换成文字，再去掉首尾空白。


# 计算一组数值的平均值；没有数值时返回 None。
def average(values):
    return sum(values) / len(values) if values else None
