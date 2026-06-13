export const API_BASE = process.env.EXPO_PUBLIC_API_URL ?? 'http://192.168.68.73:4000';
export const WS_BASE = API_BASE.replace(/^http/, 'ws');
