/// <reference types="vitest/config" />
import { defineConfig } from "vite";

export default defineConfig({
  base: "/calc-gpt/",
  build: { target: "es2022" },
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
    coverage: {
      provider: "v8",
      include: ["src/engine/**/*.ts"],
      thresholds: {
        statements: 100,
        branches: 100,
        functions: 100,
        lines: 100,
      },
      reporter: ["text", "html"],
    },
  },
});
