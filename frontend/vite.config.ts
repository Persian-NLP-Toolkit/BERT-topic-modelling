import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// In dev, /api is proxied to the FastAPI server (bert-wordcloud serve).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": process.env.VITE_API_PROXY ?? "http://127.0.0.1:8000",
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
  test: {
    environment: "node",
  },
});
