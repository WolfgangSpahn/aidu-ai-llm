/// <reference types="vite/client" />

declare module '*.css' {
  const content: { [key: string]: any };
  export default content;
}
