import { vitePreprocess } from '@sveltejs/vite-plugin-svelte'

export default {
  preprocess: vitePreprocess(),
  compilerOptions: {
    // Enable runtime checks in development
    dev: process.env.NODE_ENV !== 'production'
  }
}
