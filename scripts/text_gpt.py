import os
import httpx
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from app.core import get_config

load_dotenv()


def main():
    config=get_config()
    API_KEY = config.MODEL_API_KEY
    BASE_URL = config.MODEL_URL
    MODEL_NAME = "glm-4"  # 使用 gpt-4o，与 .env 配置一致

    try:
        # ✅ 添加超时设置
        llm = ChatOpenAI(
            model=MODEL_NAME,
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0.7,
            timeout=30,  # 30秒超时
            max_retries=2  # 最多重试2次
        )

        print(f"正在调用 {MODEL_NAME} ...")
        print(f"Base URL: {BASE_URL}")

        # ✅ 添加调试信息
        response = llm.invoke("你好，请用一句话介绍自己。")

        print("✅ 调用成功！")
        print(response.content)

    except httpx.TimeoutException:
        print("❌ 请求超时，请检查网络连接或 API 服务状态")
    except httpx.ConnectError:
        print("❌ 无法连接到服务器，请检查网络或代理设置")
    except Exception as e:
        print(f"❌ 调用失败: {type(e).__name__}")
        print(str(e))

        if "model_not_found" in str(e):
            print("\n💡 提示：模型名称可能不正确")
        elif "Invalid URL" in str(e):
            print("\n💡 提示：Base URL 格式错误")
        elif "401" in str(e) or "Unauthorized" in str(e):
            print("\n💡 提示：API Key 无效或已过期")


if __name__ == "__main__":
    main()