export const API_BASE = process.env.EXPO_PUBLIC_API_URL ?? 'http://localhost:8001';
export const WS_BASE = API_BASE.replace(/^http/, 'ws');
