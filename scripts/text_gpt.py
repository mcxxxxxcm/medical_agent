import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# 加载环境变量 (可选，建议将 key 放在 .env 文件中)
load_dotenv()


def main():
    # 【关键配置】
    API_KEY = "sk-pUfev2OwotZYatgBSqzFNzliY1wH0JVx09yqqKz8WUq87r8u"  # 替换为您的真实 Key
    BASE_URL = "https://xiaoai.plus/v1/"  # 替换为 PoloAPI 提供的实际 Base URL
    MODEL_NAME = "gpt-3.5-turbo"  # 如果报错找不到模型，尝试改为 "gpt-4-turbo" 或查看控制台的具体名称

    try:
        llm = ChatOpenAI(
            model=MODEL_NAME,
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0.7
        )

        print(f"正在调用 {MODEL_NAME} ...")
        response = llm.invoke("你是一个医疗智能助手，请用一句话进行自我介绍。")

        print("✅ 调用成功！")
        print(response.content)

    except Exception as e:
        print("❌ 调用失败:")
        print(str(e))
        if "model_not_found" in str(e):
            print("\n💡 提示：模型名称可能不正确，请检查 PoloAPI 控制台支持的模型列表。")
        elif "Invalid URL" in str(e):
            print("\n💡 提示：Base URL 格式错误，请确保不要包含 /chat/completions 后缀。")


if __name__ == "__main__":
    main()
