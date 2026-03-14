"""测试依赖安装"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    import langchain_community
    print(f"✓ langchain_community 版本: {langchain_community.__version__}")
except ImportError as e:
    print(f"✗ langchain_community 导入失败: {e}")

try:
    import langchain_core
    print(f"✓ langchain_core 版本: {langchain_core.__version__}")
except ImportError as e:
    print(f"✗ langchain_core 导入失败: {e}")

try:
    import langchain_openai
    print(f"✓ langchain_openai 版本: {langchain_openai.__version__}")
except ImportError as e:
    print(f"✗ langchain_openai 导入失败: {e}")

try:
    import langchain_chroma
    print(f"✓ langchain_chroma 版本: {langchain_chroma.__version__}")
except ImportError as e:
    print(f"✗ langchain_chroma 导入失败: {e}")

try:
    import langchain_text_splitters
    print(f"✓ langchain_text_splitters 版本: {langchain_text_splitters.__version__}")
except ImportError as e:
    print(f"✗ langchain_text_splitters 导入失败: {e}")

try:
    from app.rag import load_medical_documents
    print("✓ app.rag 模块导入成功")
except ImportError as e:
    print(f"✗ app.rag 模块导入失败: {e}")
