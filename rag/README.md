# rag 目录导航

本目录主要存放 RAG 检索、评估、测试用例、评估结果以及说明文档。

## 目录结构

### 核心代码

- `rag/rag.py`
  - 主 RAG 实现
  - 包含文档加载、切分、向量检索、`reranker` 重排、问答链构建等核心逻辑

- `rag/ragtest.py`
  - RAG 评估脚本
  - 用于批量跑测试 case，输出召回指标和答案准确率

### 测试用例

- `rag/ragtest_cases.json`
  - 当前正式评估使用的测试集
  - 每条 case 包含问题、预期来源文档、答案关键词等信息

- `rag/ragtest_cases.example.json`
  - 测试用例模板
  - 用于新增或参考 case 格式

### 评估结果

- `rag/ragtest_report.json`
  - 当前最新评估结果
  - 对应当前 `reranker` 方案

- `rag/ragtest_report.hybrid_baseline.json`
  - 旧版本评估基线
  - 用于和 `reranker` 结果做对比

- `rag/ragtest_compare.md`
  - 新旧评估结果对比报告
  - 适合直接阅读，不需要手动分析 JSON

### 说明文档

- `rag/CHANGE_SUMMARY.md`
  - 本次修改说明
  - 解释这次改了什么、为什么改、产物有什么用

- `rag/rag_py_diff_explained.md`
  - `rag.py` 改动解读
  - 面向汇报，解释从混合检索到 `reranker` 的代码级变化

- `rag/RAG_LINE_BY_LINE.md`
  - 行级说明文档
  - 用于理解 `rag.py` 各段代码的职责

- `rag/rag_annotated.py`
  - 带注释版本的 `rag.py`
  - 便于阅读实现思路

### 数据和中间产物

- `rag/papers/`
  - 原始论文 PDF 文件目录
  - 评估和检索的数据来源

- `rag/_paper_extracts/`
  - 论文抽取出的中间文本或辅助产物

- `rag/.indexed_files.json`
  - 已索引文件记录
  - 用于避免重复向量化和重复入库

### 运行缓存

- `rag/__pycache__/`
  - Python 编译缓存
  - 不需要手动维护

## 推荐阅读顺序

如果你要快速理解这个目录，建议按下面顺序看：

1. `rag/README.md`
2. `rag/ragtest_compare.md`
3. `rag/CHANGE_SUMMARY.md`
4. `rag/rag_py_diff_explained.md`
5. `rag/rag.py`
6. `rag/ragtest.py`
7. `rag/ragtest_cases.json`

## 常见用途

### 1. 看当前效果

优先看：

- `rag/ragtest_compare.md`
- `rag/ragtest_report.json`

### 2. 看本次改了什么

优先看：

- `rag/CHANGE_SUMMARY.md`
- `rag/rag_py_diff_explained.md`

### 3. 看主逻辑实现

优先看：

- `rag/rag.py`
- `rag/rag_annotated.py`
- `rag/RAG_LINE_BY_LINE.md`

### 4. 跑评估或补测试

优先看：

- `rag/ragtest.py`
- `rag/ragtest_cases.json`
- `rag/ragtest_cases.example.json`
