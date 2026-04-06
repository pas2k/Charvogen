import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

export default defineConfig({
  plugins: [svelte()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8910",
      "/ws": {
        target: "ws://localhost:8910",
        ws: true,
      },
    },
  },
});
