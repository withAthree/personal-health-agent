# web-app

Personal Health Agent 前端应用。前后端分离，通过 `/api` 与 `agent-app` 后端通信。

## 技术栈

| 技术 | 版本 | 说明 |
| --- | --- | --- |
| Node.js | `^20.19.0 \|\| >=22.12.0` | 由 Vite 8 / ESLint 10 的 engines 决定 |
| React | 19.3.0 | |
| Vite | 8.3.4 | 基于 rolldown，分包走 `codeSplitting` |
| TypeScript | 6.0.3 | 见下方「版本约束」 |
| Tailwind CSS | 4.3.3 | CSS-first，无 `tailwind.config.js` |
| ECharts | 6.1.0 | 按需注册，见 `src/lib/echarts.ts` |
| ESLint | 10.12.0 | flat config + typescript-eslint 类型感知规则 |

## 快速开始

```bash
npm install
npm run dev        # http://localhost:5173
```

其它脚本：

```bash
npm run build      # tsc -b 类型检查 + vite 生产构建
npm run preview    # 预览 dist 产物
npm run typecheck  # 仅类型检查
npm run lint       # ESLint
npm run lint:fix   # ESLint 自动修复
```

开发环境默认把 `/api` 代理到 `http://localhost:8000`（后端 FastAPI 端口，与 `document/.env.example` 的 `BACKEND_PORT` 一致），配置见 [vite.config.ts](vite.config.ts)。可复制 `.env.example` 为 `.env.local` 覆盖 `VITE_API_BASE_URL` 等变量。

## 目录结构

```
src/
├── App.tsx              # 应用外壳（页头 + 主内容区）
├── main.tsx             # 入口，挂载 #root
├── index.css            # Tailwind 引入 + @theme 设计令牌
├── vite-env.d.ts        # import.meta.env 类型
├── components/
│   └── EChart.tsx       # ECharts 容器组件（实例生命周期 + 尺寸自适应）
├── data/
│   └── demo.ts          # 演示数据，接入后端后由 services 替换
├── lib/
│   └── echarts.ts       # ECharts 按需注册 + ECOption 类型
└── pages/
    └── DashboardPage.tsx
```

## 约定

### 路径别名

`@/*` → `src/*`，在 [vite.config.ts](vite.config.ts) 与 [tsconfig.app.json](tsconfig.app.json) 两处同步定义，新增别名时需同时修改。

### 新增图表类型

`src/lib/echarts.ts` 采用按需注册以控制产物体积。新增图表或组件时，该文件内**三处需同步修改**，否则会出现类型缺失或运行时组件不生效：

1. 从 `echarts/charts`、`echarts/components` 补 import；
2. 补进 `echarts.use([...])`；
3. 把对应的 `XxxSeriesOption` / `XxxComponentOption` 加入 `ECOption` 联合类型。

### 样式

Tailwind v4 采用 CSS-first 配置：设计令牌写在 [src/index.css](src/index.css) 的 `@theme` 块中，例如 `--color-brand-600` 自动生成 `bg-brand-600` / `text-brand-600` 等工具类。项目不使用 `tailwind.config.js`。

### TypeScript

已开启 `strict`、`verbatimModuleSyntax`、`erasableSyntaxOnly`、`noUnusedLocals` / `noUnusedParameters`。因此：

- 类型导入必须写 `import type { ... }`；
- 禁用 `enum`、构造函数参数属性等非可擦除语法，用 `as const` 对象或联合类型替代。

## 版本约束

**TypeScript 固定在 `~6.0.3`，不要升到 7.x。** 截至当前，`typescript-eslint` 的 peer 范围是 `>=4.8.4 <6.1.0`，与 TypeScript 7 无法共存（`npm install` 直接 ERESOLVE）。升级 TypeScript 前需先确认 typescript-eslint 已放开该上限。

另：TypeScript 6 已废弃 `baseUrl`，`tsconfig.app.json` 中只保留 `paths`（不依赖 `baseUrl`，同时也向前兼容 TS 7）。

## 构建产物

```
dist/assets/index-*.js      ~6.7 kB    业务代码
dist/assets/react-*.js      ~219 kB    React 运行时
dist/assets/echarts-*.js    ~559 kB    ECharts（gzip ~189 kB）
```

echarts 与 react 拆为独立 chunk 以便长期缓存。`chunkSizeWarningLimit` 设为 700 kB 是因为 ECharts 按需注册后仍有约 559 kB，属预期体积而非配置疏漏。
