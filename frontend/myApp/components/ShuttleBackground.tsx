import React, { useEffect } from 'react';
import { StyleSheet, useWindowDimensions } from 'react-native';
import Animated, {
  Easing,
  SharedValue,
  useAnimatedReaction,
  useAnimatedStyle,
  useSharedValue,
  withDelay,
  withRepeat,
  withSequence,
  withSpring,
  withTiming,
} from 'react-native-reanimated';
import { Gesture, GestureDetector } from 'react-native-gesture-handler';
import Svg, { Defs, Ellipse, G, Line, Path, RadialGradient, Rect, Stop } from 'react-native-svg';

type Cfg = {
  tf: number; lf: number; size: number; opacity: number;
  ax: number; ay: number; half: number; phase: number;
};

const CONFIGS: Cfg[] = [
  { tf: 0.08, lf: 0.12, size: 26, opacity: 0.40, ax: 8,  ay: 10, half: 6000, phase: 0    },
  { tf: 0.22, lf: 0.55, size: 30, opacity: 0.50, ax: 12, ay: 14, half: 7500, phase: 900  },
  { tf: 0.12, lf: 0.72, size: 34, opacity: 0.55, ax: 10, ay: 16, half: 9000, phase: 1800 },
  { tf: 0.50, lf: 0.18, size: 38, opacity: 0.65, ax: 16, ay: 12, half: 8000, phase: 2700 },
  { tf: 0.62, lf: 0.65, size: 42, opacity: 0.75, ax: 14, ay: 20, half: 7000, phase: 3600 },
];

const REPEL_RADIUS = 120;
const REPEL_MAX = 80;
const NO_TOUCH = -9999;

function ShuttleSvg({ size }: { size: number }) {
  const h = Math.round(size * 72 / 64);
  return (
    <Svg width={size} height={h} viewBox="0 0 64 72">
      <G transform="rotate(-26 32 58)">
        <Ellipse cx={32} cy={22} rx={7} ry={16} fill="#f7f9f4" stroke="#cfd6cc" strokeWidth={1.2} />
        <Line x1={32} y1={38} x2={32} y2={55} stroke="#b9c2b6" strokeWidth={1.5} />
      </G>
      <G transform="rotate(-13 32 58)">
        <Ellipse cx={32} cy={22} rx={7} ry={16} fill="#fdfefb" stroke="#cfd6cc" strokeWidth={1.2} />
        <Line x1={32} y1={38} x2={32} y2={55} stroke="#b9c2b6" strokeWidth={1.5} />
      </G>
      <G>
        <Ellipse cx={32} cy={22} rx={7} ry={16} fill="#ffffff" stroke="#cfd6cc" strokeWidth={1.2} />
        <Line x1={32} y1={38} x2={32} y2={55} stroke="#b9c2b6" strokeWidth={1.5} />
      </G>
      <G transform="rotate(13 32 58)">
        <Ellipse cx={32} cy={22} rx={7} ry={16} fill="#f7f9f4" stroke="#cfd6cc" strokeWidth={1.2} />
        <Line x1={32} y1={38} x2={32} y2={55} stroke="#b9c2b6" strokeWidth={1.5} />
      </G>
      <G transform="rotate(26 32 58)">
        <Ellipse cx={32} cy={22} rx={7} ry={16} fill="#f0f3ee" stroke="#cfd6cc" strokeWidth={1.2} />
        <Line x1={32} y1={38} x2={32} y2={55} stroke="#b9c2b6" strokeWidth={1.5} />
      </G>
      <Rect x={23} y={54} width={18} height={6} rx={2} fill="#ffffff" stroke="#d8dcd4" strokeWidth={1} />
      <Path d="M23 60 a9 9 0 0 0 18 0 z" fill="#e3cfa4" stroke="#c9b489" strokeWidth={1} />
    </Svg>
  );
}

type ItemProps = {
  cfg: Cfg;
  touchX: SharedValue<number>;
  touchY: SharedValue<number>;
  W: number;
  H: number;
};

function ShuttleItem({ cfg, touchX, touchY, W, H }: ItemProps) {
  const driftX = useSharedValue(0);
  const driftY = useSharedValue(0);
  const repelX = useSharedValue(0);
  const repelY = useSharedValue(0);

  const baseCX = cfg.lf * W + cfg.size / 2;
  const baseCY = cfg.tf * H + cfg.size / 2;

  useEffect(() => {
    const ease = Easing.inOut(Easing.sin);
    driftX.value = withDelay(
      cfg.phase,
      withRepeat(
        withSequence(
          withTiming(cfg.ax,  { duration: cfg.half, easing: ease }),
          withTiming(-cfg.ax, { duration: cfg.half, easing: ease }),
        ),
        -1,
        false,
      ),
    );
    driftY.value = withDelay(
      cfg.phase + 300,
      withRepeat(
        withSequence(
          withTiming(cfg.ay,  { duration: cfg.half, easing: ease }),
          withTiming(-cfg.ay, { duration: cfg.half, easing: ease }),
        ),
        -1,
        false,
      ),
    );
  }, []);

  useAnimatedReaction(
    () => [touchX.value, touchY.value] as [number, number],
    ([tx, ty]) => {
      const spring = { damping: 14, stiffness: 80 };
      if (tx < NO_TOUCH + 1000) {
        repelX.value = withSpring(0, spring);
        repelY.value = withSpring(0, spring);
        return;
      }
      const ddx = baseCX - tx;
      const ddy = baseCY - ty;
      const dist = Math.sqrt(ddx * ddx + ddy * ddy);
      if (dist > 0 && dist < REPEL_RADIUS) {
        const factor = ((REPEL_RADIUS - dist) / REPEL_RADIUS) * REPEL_MAX;
        repelX.value = (ddx / dist) * factor;
        repelY.value = (ddy / dist) * factor;
      } else {
        repelX.value = withSpring(0, spring);
        repelY.value = withSpring(0, spring);
      }
    },
  );

  const style = useAnimatedStyle(() => ({
    transform: [
      { translateX: driftX.value + repelX.value },
      { translateY: driftY.value + repelY.value },
    ],
  }));

  return (
    <Animated.View
      pointerEvents="none"
      style={[styles.shuttle, { top: cfg.tf * H, left: cfg.lf * W, opacity: cfg.opacity }, style]}
    >
      <ShuttleSvg size={cfg.size} />
    </Animated.View>
  );
}

export default function ShuttleBackground({ children }: { children?: React.ReactNode }) {
  const { width: W, height: H } = useWindowDimensions();

  const touchX = useSharedValue(NO_TOUCH);
  const touchY = useSharedValue(NO_TOUCH);

  const pan = Gesture.Pan()
    .onBegin((e) => {
      'worklet';
      touchX.value = e.absoluteX;
      touchY.value = e.absoluteY;
    })
    .onUpdate((e) => {
      'worklet';
      touchX.value = e.absoluteX;
      touchY.value = e.absoluteY;
    })
    .onEnd(() => {
      'worklet';
      touchX.value = NO_TOUCH;
      touchY.value = NO_TOUCH;
    })
    .onFinalize(() => {
      'worklet';
      touchX.value = NO_TOUCH;
      touchY.value = NO_TOUCH;
    });

  return (
    <GestureDetector gesture={pan}>
      <Animated.View style={StyleSheet.absoluteFill}>
        <Svg style={StyleSheet.absoluteFill} width={W} height={H}>
          <Defs>
            <RadialGradient
              id="bgGrad"
              cx={W * 0.7}
              cy={H * 0.15}
              r={Math.max(W, H) * 1.1}
              fx={W * 0.7}
              fy={H * 0.15}
              gradientUnits="userSpaceOnUse"
            >
              <Stop offset="0%" stopColor="#14532d" />
              <Stop offset="45%" stopColor="#052e16" />
              <Stop offset="100%" stopColor="#020a05" />
            </RadialGradient>
          </Defs>
          <Rect x={0} y={0} width={W} height={H} fill="url(#bgGrad)" />
        </Svg>

        {CONFIGS.map((cfg, i) => (
          <ShuttleItem key={i} cfg={cfg} touchX={touchX} touchY={touchY} W={W} H={H} />
        ))}

        {children}
      </Animated.View>
    </GestureDetector>
  );
}

const styles = StyleSheet.create({
  shuttle: { position: 'absolute' },
});
