# 使用官方 NVIDIA CUDA 开发镜像作为基础，版本与项目要求匹配
FROM ubuntu:20.04

# 设置环境变量，避免安装过程中出现交互式提示
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR /opt/conda
# 将 Conda 添加到系统路径
ENV PATH=$CONDA_DIR/bin:$PATH

# 更新系统并安装必要的软件包 (git 用于克隆仓库, curl 用于下载 Conda)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    curl \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# --- 核心逻辑变更 ---
# 1. 首先克隆项目仓库，这样我们就可以访问到里面的 yml 文件
RUN git clone https://github.com/SantanderMetGroup/deep4downscaling.git /app

# 设置工作目录
WORKDIR /app

# 2. 安装 Miniconda
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh
    
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 3. 使用仓库中自带的 yml 文件创建 Conda 环境
# 这个过程会非常耗时
RUN conda env create -f /app/requirements/deep4downscaling-cpu.yml && \
    # 清理 Conda 缓存以减小镜像体积
    conda clean -afy

# 设置默认的 SHELL，使其自动激活新创建的 Conda 环境
SHELL ["conda", "run", "-n", "deep4downscaling-gpu", "/bin/bash", "-c"]

# 暴露 JupyterLab 端口
EXPOSE 8888

# 定义容器启动时的默认命令
# 启动 JupyterLab 服务，允许从外部访问
CMD ["jupyter-lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/app"]