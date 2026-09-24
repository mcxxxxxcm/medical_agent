"""TTFT 实测：向 /api/chat/stream 发流式请求，统计每个 SSE 事件的到达时刻。

纯标准库，不依赖项目模块。用于诊断首 token 延迟基线。
用法: python measure_ttft.py
"""
import json
import ssl
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
# 关闭代理，避免命中 WRAP_PROXY / HTTPS_PROXY
urllib.request.install_opener(urllib.request.build_opener())

QUESTIONS = [
    ("knowledge", "高血压常见的症状有哪些？"),
    ("symptom", "我最近经常头疼，需要去医院吗？"),
    ("general", "你好，在吗？"),
    ("drug", "布洛芬和感冒药可以一起吃吗？"),
]


def interpret(data: str):
    """解析 SSE data 行，返回 (type, brief)。data 已是 JSON 字符串或纯文本 token。"""
    if data.startswith("{") or data.startswith("["):
        try:
            obj = json.loads(data)
        except Exception:
            return "raw", data[:20]
        if isinstance(obj, dict) and "type" in obj:
            t = obj["type"]
            if t == "sources":
                names = [s.get("source", "") for s in obj.get("sources", [])]
                return "sources", ",".join(names)[:40]
            if t == "done":
                return "done", f'elapsed={obj.get("total_elapsed_ms")}ms'
            if t == "status":
                return "status", f'({obj.get("status", "?")})'
            return t, str(obj)[:30]
        return "token", data[:20]
    return "token", data[:20]


def measure(qid, question, rounds=2):
    print(f"\n{'='*70}\n[{qid}] {question}")
    results = []
    for r in range(rounds):
        body = json.dumps({"question": question, "user_id": "ttft-test", "thread_id": f"{qid}-{r}"})
        req = urllib.request.Request(
            BASE + "/api/chat/stream",
            data=body.encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            method="POST",
        )
        t0 = time.time()
        first_token = None
        first_event = None
        first_status = None
        done_at = None
        last_token_at = first_token
        n_token = 0
        try:
            ctx = ssl._create_unverified_context()
            with urllib.request.urlopen(req, timeout=90, context=ctx) as resp:
                n_tokens = 0
                buf = ""
                for raw_line in resp:
                    line = raw_line.decode("utf-8", "ignore").rstrip("\r\n")
                    t = (time.time() - t0) * 1000
                    if line.startswith("data:"):
                        payload = line[5:].strip()
                        if not payload:
                            continue
                        kind, _ = interpret(payload)
                        if kind == "status" and first_status is None:
                            first_status = t
                        elif kind in ("token", "answer_chunk"):
                            n_tokens += 1
                            if first_token is None:
                                first_token = t
                        elif kind == "sources":
                            if first_event is None:
                                first_event = t
                        elif kind == "done":
                            done_at = t
                    # SSE 首行边界即首事件
                    if first_event is None:
                        # data 行本身即首事件
                        first_event = first_token
        except Exception as e:
            print(f"  round{r} 异常: {e}")
            continue
        print(
            f"  round{r}: status={first_status or 0:.0f}ms 首事件={first_event or 0:.0f}ms "
            f"首TOKEN={first_token or 0:.0f}ms done={done_at or 0:.0f}ms token数={n_tokens}"
        )
        results.append(first_token)
    return results


if __name__ == "__main__":
    for qid, q in QUESTIONS:
        measure(qid, q)