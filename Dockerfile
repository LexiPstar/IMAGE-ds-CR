# 使用更小的基础镜像 + 中文字体支持
FROM python:3.9-slim

WORKDIR /app
# 安装依赖
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
# 复制代码

COPY . .
# 暴露端口
EXPOSE 8000

# 启动应用，监听所有 IP 地址
CMD ["uvicorn", "inferences.inference:app", "--host", "0.0.0.0", "--port", "8000"]
