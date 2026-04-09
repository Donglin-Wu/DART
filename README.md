# DART 代码仓库

这是一个只保留代码的整理版本，适合直接作为 GitHub 代码仓库上传。

## 目录

```text
git/
├─ pyproject.toml
├─ src/dart/      # 核心源码
├─ scripts/       # 运行脚本与数据处理脚本
├─ README.md
├─ requirements.txt
├─ .gitignore
└─ .gitattributes
```

## 说明

- `src/dart/` 中是核心算法代码。
- `scripts/batch/` 中是批处理脚本。
- `scripts/one_dimensional/` 中是一维实验脚本。
- `scripts/data_processing/` 中是数据处理脚本。
- 当前仓库不包含任何输入数据、验证数据或计算结果。

## 依赖安装

```powershell
pip install -r requirements.txt
```

如果要按标准包方式使用，可以在仓库根目录执行：

```powershell
pip install -e .
```

## 注意

- 脚本只保留代码本身。
- 如果要实际运行，需要自行补充对应的数据目录和输入文件。
