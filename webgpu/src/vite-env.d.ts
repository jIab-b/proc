/// <reference types="vite/client" />

interface ImportMetaEnv {
  // No environment variables needed - API keys provided by user
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
