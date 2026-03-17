import os
from pathlib import Path
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

# 1. 定义路径 (建议使用绝对路径)
# 假设你的脚本在 scripts/ 目录下，模型在上一级目录，或者你直接写绝对路径
script_dir = Path(__file__).parent
# 这里根据你的实际目录结构调整，假设模型文件夹在 project 根目录下
model_path = script_dir.parent / "bge-reranker-v2-m3"
output_path = script_dir.parent / "bge-reranker-onnx"

# 转换为绝对路径字符串，避免相对路径引发的解析错误
model_id = str(model_path.resolve())
output_dir = str(output_path.resolve())

print(f"正在从本地路径加载模型: {model_id}")
print(f"输出目录: {output_dir}")

# 2. 加载 Tokenizer (先验证本地模型是否有效)
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    print("✅ Tokenizer 加载成功")
except Exception as e:
    print(f"❌ Tokenizer 加载失败，请检查路径是否正确: {e}")
    exit(1)

# 3. 加载并导出模型
# 关键点:
# - export=True: 强制从 transformers 导出
# - task="sequence-classification": 明确任务类型 (Reranker 本质是分类)
# - local_files_only=True (如果可用): 防止联网尝试 (但在 from_pretrained 中可能不支持此参数，主要靠绝对路径和 export=True)
try:
    model = ORTModelForSequenceClassification.from_pretrained(
        model_id,
        export=True,  # 强制导出
        # opset=14               # 可选：指定 ONNX opset 版本，通常 14 或 17 比较稳
    )

    print("✅ 模型加载/导出成功，正在保存...")

    # 4. 保存模型
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"🎉 完成！ONNX 模型已保存至: {output_dir}")

except Exception as e:
    print(f"❌ 转换过程中出错: {e}")
    # 如果是网络相关的报错，说明它还在尝试联网，请检查模型路径下是否有 config.json
    import traceback

    traceback.print_exc()