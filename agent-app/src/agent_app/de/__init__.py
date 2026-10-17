"""DE 子 Agent：医学证据分析员。

它是 ``create_agent`` 装配出来的独立 agent，通过 ``de/tool.py`` 的 ``de_expert``
挂到主 agent 上。与 ``ds/`` 的区别：DS 用手写 StateGraph 是因为它有 Plan / Code /
Execute / Check 四个阶段和沙箱；DE 的循环就是 ReAct，由框架提供，不需要重写。
"""
