// Playwright type declarations for development
// This provides basic types when playwright package is not installed

export interface Page {
  goto(url: string, options?: any): Promise<any>;
  locator(selector: string): Locator;
  evaluate<T>(pageFunction: () => T): Promise<T>;
  evaluate<T, Arg>(pageFunction: (arg: Arg) => T, arg: Arg): Promise<T>;
  on(event: string, handler: Function): void;
  close(): Promise<void>;
  newPage(): Promise<Page>;
  waitForLoadState(state?: string, options?: any): Promise<void>;
  querySelectorAll(selector: string): Promise<any[]>;
  addInitScript(script: Function): Promise<void>;
  waitForTimeout(timeout: number): Promise<void>;
  keyboard: {
    press(key: string): Promise<void>;
  };
}

export interface Locator {
  all(): Promise<any[]>;
  click(): Promise<void>;
  fill(text: string): Promise<void>;
  textContent(): Promise<string | null>;
}

export interface Browser {
  newPage(): Promise<Page>;
  close(): Promise<void>;
}

export interface BrowserContext {
  newPage(): Promise<Page>;
  close(): Promise<void>;
}