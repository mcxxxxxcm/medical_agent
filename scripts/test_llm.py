from app.core.llm import get_llm


def main():
    llm = get_llm()
    resp = llm.invoke('你是一个医疗智能助手，请用一句话进行自我介绍。')
    print(resp)


if __name__ == '__main__':
    main()
