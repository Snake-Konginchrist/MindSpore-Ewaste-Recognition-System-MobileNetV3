# UV包管理器使用说明

本项目使用UV作为Python包管理器，并指定Python版本为3.11。

## 安装UV

### macOS/Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## 项目设置

### 1. 初始化项目
```bash
# 创建虚拟环境并安装依赖
uv sync
```

### 2. 激活虚拟环境
```bash
# 激活虚拟环境
source .venv/bin/activate  # macOS/Linux
# 或
.venv\Scripts\activate     # Windows
```

### 3. 安装依赖
```bash
# 安装所有依赖（包括开发依赖）
uv sync --dev

# 仅安装生产依赖
uv sync

# 添加新依赖
uv add package-name

# 添加开发依赖
uv add --dev package-name
```

### 4. 运行项目
```bash
# 运行主程序
python ewaste_recognition.py

# 运行数据预处理
python data_processing/main.py

# 运行训练
python core/train.py
```

## 常用命令

```bash
# 查看已安装的包
uv pip list

# 更新依赖
uv sync --upgrade

# 移除依赖
uv remove package-name

# 运行测试
uv run pytest

# 代码格式化
uv run black .

# 类型检查
uv run mypy .

# 代码检查
uv run flake8 .
```

## 配置文件说明

- `pyproject.toml`: 项目主要配置文件，包含依赖、构建配置等
- `.uv.toml`: UV专用配置文件，指定Python版本和项目设置
- `uv.lock`: 依赖版本锁定文件（自动生成）

## Python版本要求

本项目要求Python版本为3.11或更高版本。UV会自动下载并管理指定版本的Python解释器。

## 注意事项

1. 确保系统已安装UV包管理器
2. 首次运行`uv sync`时会自动下载Python 3.11
3. 虚拟环境位于`.venv`目录
4. 依赖缓存位于`.uv-cache`目录
5. 不要手动编辑`uv.lock`文件，它由UV自动维护 