


pandoc  -t latex '第 3 章：Prompt 提示词工程最佳实践与效果评估优化.md'  | pandoc -f latex --data-dir=docs/rendering/ -o 3.docx
pandoc  -t latex '第 4 章 RAG 检索增强生成.md'  | pandoc -f latex --data-dir=docs/rendering/ -o 4.docx

