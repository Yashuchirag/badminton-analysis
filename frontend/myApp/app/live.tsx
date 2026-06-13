import { useEffect, useRef, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { CameraView, useCameraPermissions, useMicrophonePermissions } from 'expo-camera';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withSpring,
} from 'react-native-reanimated';
import * as FileSystem from 'expo-file-system/legacy';
import { useRouter } from 'expo-router';

import { API_BASE, WS_BASE } from '@/lib/config';
import { colors, fonts, spacing } from '@/lib/theme';
import ShuttleBackground from '@/components/ShuttleBackground';

const SEGMENT_SECONDS = 3;
const MAX_UPLOADS_IN_FLIGHT = 2;

interface LiveState {
  status: string;
  score: [number, number];
  rally_state: string;
  last_hitter_side: string | null;
  backlog: number;
  total_landings?: number;
  message?: string;
}

type Phase = 'idle' | 'starting' | 'running' | 'stopping' | 'done';

export default function LiveScreen() {
  const router = useRouter();
  const [cameraPermission, requestCameraPermission] = useCameraPermissions();
  const [micPermission, requestMicPermission] = useMicrophonePermissions();

  const [phase, setPhase] = useState<Phase>('idle');
  const [liveState, setLiveState] = useState<LiveState | null>(null);
  const [connection, setConnection] = useState<'ws' | 'polling' | 'none'>('none');
  const [error, setError] = useState<string | null>(null);

  const cameraRef = useRef<CameraView>(null);
  const sessionIdRef = useRef<string | null>(null);
  const runningRef = useRef(false);
  const seqRef = useRef(0);
  const queueRef = useRef<{ seq: number; uri: string }[]>([]);
  const inFlightRef = useRef(0);
  const wsRef = useRef<WebSocket | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const phaseRef = useRef<Phase>('idle');
  phaseRef.current = phase;

  const calBannerY = useSharedValue(-80);
  const calBannerStyle = useAnimatedStyle(() => ({
    transform: [{ translateY: calBannerY.value }],
  }));

  const calibrating =
    phase === 'running' &&
    (!liveState || ['starting', 'loading', 'calibrating'].includes(liveState.status));

  useEffect(() => {
    calBannerY.value = withSpring(calibrating ? 0 : -80, { damping: 14, stiffness: 80 });
  }, [calibrating]);

  useEffect(() => {
    return () => {
      runningRef.current = false;
      cameraRef.current?.stopRecording();
      wsRef.current?.close();
      stopPolling();
    };
  }, []);

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  const handleState = (state: LiveState) => {
    setLiveState(state);
    if (state.status === 'complete' || state.status === 'error') {
      stopPolling();
      wsRef.current?.close();
      wsRef.current = null;
      if (state.status === 'error') {
        setError(state.message || 'Session failed');
      }
      setPhase('done');
    }
  };

  const startPolling = () => {
    if (pollRef.current || !sessionIdRef.current) return;
    setConnection('polling');
    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/live/status/${sessionIdRef.current}`);
        if (res.ok) handleState(await res.json());
      } catch {
        // server unreachable; keep trying
      }
    }, 1000);
  };

  const openWebSocket = (sessionId: string) => {
    const ws = new WebSocket(`${WS_BASE}/live/ws/${sessionId}`);
    wsRef.current = ws;
    ws.onopen = () => setConnection('ws');
    ws.onmessage = (e) => {
      try {
        handleState(JSON.parse(e.data));
      } catch {
        // ignore malformed message
      }
    };
    const fallBack = () => {
      if (wsRef.current === ws) wsRef.current = null;
      if (phaseRef.current === 'running' || phaseRef.current === 'stopping') {
        startPolling();
      }
    };
    ws.onerror = fallBack;
    ws.onclose = fallBack;
  };

  const uploadChunk = async (item: { seq: number; uri: string }) => {
    const formData = new FormData();
    formData.append('file', {
      uri: item.uri,
      name: `chunk_${item.seq}.mp4`,
      type: 'video/mp4',
    } as any);
    try {
      await fetch(`${API_BASE}/live/chunk/${sessionIdRef.current}?seq=${item.seq}`, {
        method: 'POST',
        body: formData,
        headers: { 'Content-Type': 'multipart/form-data' },
      });
    } catch (err) {
      console.warn('Chunk upload failed:', err);
    } finally {
      FileSystem.deleteAsync(item.uri, { idempotent: true }).catch(() => {});
    }
  };

  const pumpUploads = () => {
    while (inFlightRef.current < MAX_UPLOADS_IN_FLIGHT && queueRef.current.length > 0) {
      const item = queueRef.current.shift()!;
      inFlightRef.current += 1;
      uploadChunk(item).finally(() => {
        inFlightRef.current -= 1;
        pumpUploads();
      });
    }
  };

  const enqueueChunk = (uri: string) => {
    queueRef.current.push({ seq: seqRef.current++, uri });
    pumpUploads();
  };

  const recordLoop = async () => {
    while (runningRef.current) {
      try {
        const video = await cameraRef.current?.recordAsync({ maxDuration: SEGMENT_SECONDS });
        if (video?.uri) enqueueChunk(video.uri);
      } catch (err) {
        if (runningRef.current) {
          console.warn('Recording error:', err);
          await new Promise((r) => setTimeout(r, 500));
        }
      }
    }
  };

  const start = async () => {
    setError(null);
    if (!cameraPermission?.granted) {
      const res = await requestCameraPermission();
      if (!res.granted) return;
    }
    if (!micPermission?.granted) {
      const res = await requestMicPermission();
      if (!res.granted) return;
    }

    setPhase('starting');
    try {
      const res = await fetch(`${API_BASE}/live/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: 'singles', annotate: false, detect_stride: 2 }),
      });
      if (!res.ok) {
        const detail = (await res.json().catch(() => null))?.detail;
        throw new Error(detail || 'Could not start session');
      }
      const data = await res.json();
      sessionIdRef.current = data.session_id;
      seqRef.current = 0;
      queueRef.current = [];
      setLiveState(null);

      openWebSocket(data.session_id);
      runningRef.current = true;
      setPhase('running');
      recordLoop();
    } catch (err: any) {
      setError(err.message || 'Could not start session');
      setPhase('idle');
    }
  };

  const stop = async () => {
    setPhase('stopping');
    runningRef.current = false;
    cameraRef.current?.stopRecording();

    const deadline = Date.now() + 15000;
    while ((queueRef.current.length > 0 || inFlightRef.current > 0) && Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, 200));
    }

    try {
      await fetch(`${API_BASE}/live/finish/${sessionIdRef.current}`, { method: 'POST' });
    } catch {
      // status updates will surface the outcome
    }
    if (!wsRef.current) startPolling();
  };

  const reset = () => {
    sessionIdRef.current = null;
    setLiveState(null);
    setError(null);
    setConnection('none');
    setPhase('idle');
  };

  if (!cameraPermission) {
    return <View style={styles.root} />;
  }

  const score = liveState?.score ?? [0, 0];

  // Done state: frosted card over shuttle background
  if (phase === 'done') {
    return (
      <View style={styles.root}>
        <ShuttleBackground>
        <SafeAreaView style={styles.safe}>
          <View style={styles.doneCard}>
            <Text style={styles.doneTitle}>Session Complete</Text>
            <View style={styles.doneScoreRow}>
              <Text style={styles.doneScoreGreen}>{score[0]}</Text>
              <View style={styles.scoreDivider} />
              <Text style={styles.doneScoreWhite}>{score[1]}</Text>
            </View>
            <Text style={styles.monoLabel}>
              {liveState?.total_landings ?? 0} LANDINGS DETECTED
            </Text>
            {error && <Text style={styles.errorText}>{error}</Text>}
            <TouchableOpacity style={styles.ctaButton} onPress={reset} activeOpacity={0.85}>
              <Text style={styles.ctaText}>New Session</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.backButton} onPress={() => router.back()} activeOpacity={0.85}>
              <Text style={styles.backText}>Back to Home</Text>
            </TouchableOpacity>
          </View>
        </SafeAreaView>
        </ShuttleBackground>
      </View>
    );
  }

  // Idle / starting state: shuttle background with centered start card
  if (phase === 'idle' || phase === 'starting') {
    return (
      <View style={styles.root}>
        <ShuttleBackground>
        <SafeAreaView style={styles.safe}>
          <View style={styles.idleCard}>
            <Ionicons name="radio-outline" size={40} color={colors.accent} />
            <Text style={styles.idleTitle}>Live Scoring</Text>
            <Text style={styles.idleSubtitle}>Point the camera at the court and tap Start.</Text>
            {error && <Text style={styles.errorText}>{error}</Text>}
            {phase === 'idle' ? (
              <TouchableOpacity style={styles.ctaButton} onPress={start} activeOpacity={0.85}>
                <Ionicons name="radio-button-on" size={20} color={colors.ctaText} />
                <Text style={styles.ctaText}>Start session</Text>
              </TouchableOpacity>
            ) : (
              <View style={styles.ctaButton}>
                <ActivityIndicator color={colors.ctaText} />
                <Text style={styles.ctaText}>Starting…</Text>
              </View>
            )}
            <TouchableOpacity style={styles.backButton} onPress={() => router.back()} activeOpacity={0.85}>
              <Text style={styles.backText}>Back</Text>
            </TouchableOpacity>
          </View>
        </SafeAreaView>
        </ShuttleBackground>
      </View>
    );
  }

  // Running / stopping: camera feed with overlay
  const connColor =
    connection === 'ws' ? colors.success :
    connection === 'polling' ? colors.warning :
    colors.destructive;

  const connLabel =
    connection === 'ws' ? 'LIVE' :
    connection === 'polling' ? 'POLLING' :
    'CONNECTING';

  return (
    <View style={styles.root}>
      <ShuttleBackground />
      <CameraView ref={cameraRef} style={StyleSheet.absoluteFill} facing="back" mode="video" />

      <SafeAreaView style={styles.cameraOverlay} pointerEvents="box-none">

        {/* Top glass strip with scores */}
        <View style={styles.topStrip} pointerEvents="none">
          <View style={styles.scoreRow}>
            <Text style={styles.scoreGreen}>{score[0]}</Text>
            <View style={styles.scoreDivider} />
            <Text style={styles.scoreWhite}>{score[1]}</Text>
          </View>
          {liveState?.rally_state ? (
            <Text style={styles.rallyLabel}>
              {liveState.rally_state === 'RALLY_ACTIVE' ? 'RALLY IN PLAY' : 'WAITING FOR SERVE'}
            </Text>
          ) : null}
        </View>

        {/* Calibrating banner */}
        <Animated.View style={[styles.calBanner, calBannerStyle]} pointerEvents="none">
          <Ionicons name="scan-outline" size={15} color="#000" />
          <Text style={styles.calText}>Calibrating, keep court in frame</Text>
        </Animated.View>

        <View style={styles.spacer} pointerEvents="none" />

        {/* Bottom row */}
        <View style={styles.bottomRow}>
          <View style={styles.connBadge} pointerEvents="none">
            <View style={[styles.connDot, { backgroundColor: connColor }]} />
            <Text style={styles.connLabel}>{connLabel}</Text>
            {liveState && liveState.backlog > 0 ? (
              <Text style={styles.connLabel}>  {liveState.backlog * SEGMENT_SECONDS}s lag</Text>
            ) : null}
          </View>

          {phase === 'running' ? (
            <TouchableOpacity style={styles.stopCircle} onPress={stop} activeOpacity={0.85}>
              <Ionicons name="stop" size={28} color="#fff" />
            </TouchableOpacity>
          ) : (
            <View style={styles.stopCircle}>
              <ActivityIndicator color="#fff" />
            </View>
          )}
        </View>

      </SafeAreaView>
    </View>
  );
}

const GLASS = {
  backgroundColor: colors.surface,
  borderWidth: 1,
  borderColor: colors.surfaceBorder,
  borderRadius: 20,
};

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: colors.background },
  safe: { flex: 1, justifyContent: 'center', paddingHorizontal: spacing(2.5) },

  // Idle / starting
  idleCard: {
    ...GLASS,
    alignItems: 'center',
    padding: spacing(3),
    gap: spacing(1.5),
  },
  idleTitle: {
    fontFamily: fonts.heading,
    fontSize: 36,
    color: colors.textPrimary,
    letterSpacing: 0.5,
  },
  idleSubtitle: {
    fontFamily: fonts.body,
    fontSize: 15,
    color: colors.textMuted,
    textAlign: 'center',
  },

  ctaButton: {
    flexDirection: 'row',
    height: 52,
    borderRadius: 14,
    backgroundColor: colors.ctaBackground,
    borderWidth: 1,
    borderColor: colors.ctaBorder,
    justifyContent: 'center',
    alignItems: 'center',
    gap: spacing(1),
    paddingHorizontal: spacing(4),
    marginTop: spacing(1),
    alignSelf: 'stretch',
  },
  ctaText: {
    fontFamily: fonts.bodyBold,
    fontSize: 17,
    color: colors.ctaText,
  },
  backButton: { padding: spacing(1), alignSelf: 'center' },
  backText: { fontFamily: fonts.bodyMedium, fontSize: 15, color: colors.textMuted },

  errorText: {
    fontFamily: fonts.body,
    fontSize: 14,
    color: colors.destructive,
    textAlign: 'center',
  },

  // Done
  doneCard: {
    ...GLASS,
    alignItems: 'center',
    padding: spacing(3),
    gap: spacing(1.5),
  },
  doneTitle: {
    fontFamily: fonts.headingMedium,
    fontSize: 26,
    color: colors.textPrimary,
  },
  doneScoreRow: { flexDirection: 'row', alignItems: 'center', gap: spacing(2) },
  doneScoreGreen: {
    fontFamily: fonts.heading,
    fontSize: 88,
    color: colors.accent,
    fontVariant: ['tabular-nums'],
  },
  doneScoreWhite: {
    fontFamily: fonts.heading,
    fontSize: 88,
    color: colors.textPrimary,
    fontVariant: ['tabular-nums'],
  },

  monoLabel: {
    fontFamily: fonts.mono,
    fontSize: 10,
    color: colors.textMuted,
    letterSpacing: 2,
    textAlign: 'center',
  },

  // Running overlay
  cameraOverlay: { flex: 1, paddingHorizontal: spacing(2) },

  topStrip: {
    ...GLASS,
    borderRadius: 16,
    paddingHorizontal: spacing(2.5),
    paddingVertical: spacing(1.5),
    marginTop: spacing(1),
    alignItems: 'center',
  },
  scoreRow: { flexDirection: 'row', alignItems: 'center', gap: spacing(2) },
  scoreGreen: {
    fontFamily: fonts.heading,
    fontSize: 72,
    color: colors.accent,
    fontVariant: ['tabular-nums'],
  },
  scoreWhite: {
    fontFamily: fonts.heading,
    fontSize: 72,
    color: colors.textPrimary,
    fontVariant: ['tabular-nums'],
  },
  scoreDivider: { width: 1, height: 56, backgroundColor: colors.surfaceBorder },
  rallyLabel: {
    fontFamily: fonts.mono,
    fontSize: 10,
    color: colors.textMuted,
    letterSpacing: 2,
    marginTop: 2,
  },

  calBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing(1),
    backgroundColor: colors.warning,
    borderRadius: 20,
    paddingHorizontal: spacing(2),
    paddingVertical: 7,
    alignSelf: 'center',
    marginTop: spacing(1),
  },
  calText: { fontFamily: fonts.bodyMedium, fontSize: 13, color: '#000' },

  spacer: { flex: 1 },

  bottomRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingBottom: spacing(2),
  },
  connBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: colors.overlayDark,
    borderRadius: 12,
    paddingHorizontal: spacing(1.5),
    paddingVertical: 6,
  },
  connDot: { width: 8, height: 8, borderRadius: 4 },
  connLabel: {
    fontFamily: fonts.mono,
    fontSize: 10,
    color: colors.textPrimary,
    letterSpacing: 1,
  },

  stopCircle: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: colors.destructive,
    alignItems: 'center',
    justifyContent: 'center',
  },
});
