"""长期记忆：抽取、存储与注入。

三条路径互相独立：

* ``store`` —— 只有它碰数据库，**所有查询都必须带 ``user_id``**；
* ``extract`` —— 每轮对话结束后后台跑一次，把值得记的事写进 ``store``；
* ``middleware`` —— 每次模型调用前，把 ``store`` 里的记忆渲染成一段 Markdown
  临时注入 system prompt（不进 checkpoint、不进账本）。

设计取舍见 ``document/模型使用清单.md`` 与 ``document/设计文档.md`` §3.3。
"""
