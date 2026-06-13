import { Platform } from 'react-native';

export const colors = {
  background: '#020a05',
  surface: 'rgba(255,255,255,0.07)',
  surfaceBorder: 'rgba(255,255,255,0.18)',
  surfacePressed: 'rgba(255,255,255,0.12)',
  textPrimary: '#ffffff',
  textMuted: '#86efac',
  accent: '#4ade80',
  accentDim: 'rgba(74,222,128,0.18)',
  accentBorder: 'rgba(74,222,128,0.45)',
  ctaBackground: '#16a34a',
  ctaBorder: '#4ade80',
  ctaText: '#ffffff',
  destructive: '#dc2626',
  warning: '#fbbf24',
  success: '#4ade80',
  overlayDark: 'rgba(2,10,5,0.72)',
};

export const fonts = {
  heading: 'BarlowCondensed_700Bold',
  headingMedium: 'BarlowCondensed_500Medium',
  body: 'Barlow_400Regular',
  bodyMedium: 'Barlow_500Medium',
  bodyBold: 'Barlow_600SemiBold',
  mono: Platform.select({ ios: 'Menlo', android: 'monospace', default: 'monospace' }) as string,
};

export const spacing = (n: number) => n * 8;
