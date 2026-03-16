# ============================================
# Medical Assistant - Dockerfile
# 基于 Python 3.11 的医疗助手应用镜像
# ============================================

# 使用官方 Python 镜像作为基础镜像

# 设置工作目录
FROM python:3.11-slim

# ---------------------------------------------------------
# 【关键修改】在运行 apt-get 之前，先替换为阿里云 Debian 源
# 针对 Debian Trixie (Testing/Unstable) 或 Bookworm (Stable)
# ---------------------------------------------------------
RUN sed -i 's|http://deb.debian.org|https://mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources \
    && sed -i 's|http://security.debian.org|https://mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources \
    || echo "Debian sources list not found in default location, trying legacy..." \
    && if [ -f /etc/apt/sources.list ]; then sed -i 's|http://deb.debian.org|https://mirrors.aliyun.com|g' /etc/apt/sources.list; fi


# 设置工作目录
WORKDIR /app

# 设置环境变量
# PYTHONDONTWRITEBYTECODE: 防止 Python 生成 .pyc 文件
# PYTHONUNBUFFERED: 确保 Python 输出直接显示在终端
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖
# build-essential: 编译某些 Python 包所需
# libpq-dev: PostgreSQL 客户端库
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY app/ ./app/
COPY data/ ./data/
COPY docs/ ./docs/
COPY scripts/ ./scripts/

# 创建日志目录与外挂reranker模型volume卷
RUN mkdir -p logs /app/models




# 暴露端口
EXPOSE 8000

# 健康检查
# 每 30 秒检查一次服务是否正常运行
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# 启动命令
# 使用 uvicorn 运行 FastAPI 应用
CMD ["uvicorn", "app.api.routes:app", "--host", "0.0.0.0", "--port", "8000"]