import { fileURLToPath, URL } from 'node:url'

import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
// `vitest/config` 的 defineConfig 是 vite 那个的超集：多一个 `test` 字段，
// 其余（`plugins` / `resolve` / `build`）完全一致，所以测试与构建共用一个别名表。
import { defineConfig } from 'vitest/config'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  test: {
    // 组件要真实 DOM：jsdom 提供 document / window 与 localStorage（api.ts 读 token）
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
    // 每个用例后自动还原 spy 与 stubGlobal，避免用例之间互相污染
    restoreMocks: true,
    unstubGlobals: true,
  },
  server: {
    port: 5173,
    proxy: {
      // 后端 FastAPI 端口，与 document/.env.example 的 BACKEND_PORT 保持一致
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    // echarts 已按需注册，但 line/bar/gauge + core 仍有约 560 kB（gzip 约 190 kB）。
    // 该块独立缓存且不随业务代码变动，故放宽告警阈值而非强行拆散。
    chunkSizeWarningLimit: 700,
    // Vite 8 基于 rolldown，分包使用 codeSplitting.groups（rollupOptions 已废弃）
    rolldownOptions: {
      output: {
        codeSplitting: {
          groups: [
            // echarts 体积大且随图表类型增减，单独成块便于浏览器长期缓存
            { name: 'echarts', test: /node_modules[\\/]echarts[\\/]/ },
            { name: 'react', test: /node_modules[\\/](react|react-dom|scheduler)[\\/]/ },
          ],
        },
      },
    },
  },
})
