/*
 * Copyright (C) 2026 Dr. Wolfgang Spahn, PHBern
 *
 * MIT License — see LICENSE file for details.
 * If you use this software in academic work, citation of the original author is requested.
 */
/// <reference types="vite/client" />

declare module '*.css' {
  const content: { [key: string]: any };
  export default content;
}
