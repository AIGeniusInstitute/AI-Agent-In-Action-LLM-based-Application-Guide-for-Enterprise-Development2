


pandoc  -t latex '第3章 Prompt 提示词工程最佳实践与效果评估优化.md'  | pandoc -f latex --data-dir=docs/rendering/ -o 3.docx
pandoc  -t latex '第4章 RAG 检索增强生成.md'  | pandoc -f latex --data-dir=docs/rendering/ -o 4.docx
pandoc  -t latex '第5章 AI Agent 应用架构设计模式与应用.md'  | pandoc -f latex --data-dir=docs/rendering/ -o 5.docx
pandoc  -t latex '第6章 AI Agent 应用开发实践.md'  | pandoc -f latex --data-dir=docs/rendering/ -o 6.docx
pandoc  -t latex '第7章 Agent应用评测优化与运维监控.md'  | pandoc -f latex --data-dir=docs/rendering/ -o 7.docx
pandoc  -t latex '第8章 Multi-Agent 系统架构设计与应用.md'  | pandoc -f latex --data-dir=docs/rendering/ -o 8.docx
pandoc  -t latex '第9章 AI Agent 工作流设计与应用场景'  | pandoc -f latex --data-dir=docs/rendering/ -o 9.docx
pandoc  -t latex '第10章 AI Agent未来发展趋势与挑战.md'  | pandoc -f latex --data-dir=docs/rendering/ -o 10.docx
