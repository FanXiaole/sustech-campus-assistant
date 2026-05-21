# SUSTech Campus RAG Knowledge Base System

基于检索增强生成（RAG）技术的南方科技大学校园知识库问答系统。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
export SILICONFLOW_API_KEY="your-key-here"

# 3. 运行端到端 Pipeline
bash scripts/run_all.sh

# 4. 启动 Demo
python demo/app.py
```

## 项目结构

```
sustech-rag/
├── config.py              # 全局配置
├── data/                  # 数据目录
│   ├── raw/               # 原始爬虫数据
│   ├── processed/         # 清洗后文档
│   ├── chunks/            # Chunk 文件
│   └── test_set.json      # 评测问题集
├── indexing/              # 数据采集与索引
├── retrieval/             # 检索管线
├── generation/            # LLM 生成与人格定制
├── evaluation/            # 评测框架
├── demo/                  # Gradio 演示
└── scripts/               # 自动化脚本
```

## 创新点

1. **HyDE** — 假设性文档嵌入查询扩展
2. **Hybrid RRF** — 稠密+稀疏 Reciprocal Rank Fusion
3. **Contextual Enrichment** — 基于 LLM 的上下文增强
4. **Query Classifier** — 查询类型动态路由
5. **Source Authority Scoring** — 来源权威性排序信号
6. **Confidence Abstention** — 证据不足时拒答
7. **🎭 人格定制** — 5 种可切换的回答风格（标准/疯狂/魅惑/闲聊/学术）

详见 [REPORT.md](REPORT.md)
