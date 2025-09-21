const isBrowser = typeof window !== 'undefined';

const DEFAULT_API_PORT = 3001;
const DEFAULT_WS_PORT = 8765;

const sanitizeUrl = (url: string) => {
  if (!url || !url.endsWith('/')) {
    return url;
  }
  return url.endsWith('://') ? url : url.slice(0, -1);
};

const resolveHostname = () => {
  if (!isBrowser) {
    return 'localhost';
  }
  return window.location.hostname || 'localhost';
};

const resolveHttpProtocol = () => {
  if (!isBrowser) {
    return 'http:';
  }
  return window.location.protocol === 'https:' ? 'https:' : 'http:';
};

const resolveWsProtocol = () => {
  if (!isBrowser) {
    return 'ws:';
  }
  return window.location.protocol === 'https:' ? 'wss:' : 'ws:';
};

const getEnvValue = (key: string): string | undefined => {
  const value = import.meta.env?.[key as keyof ImportMetaEnv];
  if (typeof value === 'string' && value.trim().length > 0) {
    return value.trim();
  }
  return undefined;
};

const defaultApiBaseUrl = () => {
  const hostname = resolveHostname();
  const protocol = resolveHttpProtocol();
  return `${protocol}//${hostname}:${DEFAULT_API_PORT}`;
};

const defaultWebSocketUrl = () => {
  const hostname = resolveHostname();
  const protocol = resolveWsProtocol();
  return `${protocol}//${hostname}:${DEFAULT_WS_PORT}`;
};

export const API_BASE_URL = sanitizeUrl(
  getEnvValue('VITE_API_BASE_URL') ?? defaultApiBaseUrl()
);

export const WEBSOCKET_URL = getEnvValue('VITE_WS_URL') ?? defaultWebSocketUrl();

export const getResolvedApiBaseUrl = () => API_BASE_URL;
export const getResolvedWebSocketUrl = () => WEBSOCKET_URL;
