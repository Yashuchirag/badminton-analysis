import { useState, useRef, useEffect } from 'react';
import {
  StyleSheet,
  ActivityIndicator,
  TouchableOpacity,
  View,
  ScrollView,
  Alert,
  Text,
  Animated,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { Video, ResizeMode } from 'expo-av';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';

import { API_BASE } from '@/lib/config';
import { colors, fonts, spacing } from '@/lib/theme';
import ShuttleBackground from '@/components/ShuttleBackground';

export default function HomeScreen() {
  const router = useRouter();
  const [videoUri, setVideoUri] = useState<string | null>(null);
  const [processing, setProcessing] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [totalFrames, setTotalFrames] = useState(0);
  const [progress, setProgress] = useState(0);
  const [score, setScore] = useState<[number, number]>([0, 0]);
  const [rallyState, setRallyState] = useState<string>('');
  const [finalResult, setFinalResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const progressAnim = useRef(new Animated.Value(0)).current;
  const pollingInterval = useRef<number | null>(null);

  useEffect(() => {
    if (jobId && processing) {
      pollingInterval.current = setInterval(async () => {
        try {
          const response = await fetch(`${API_BASE}/job-status/${jobId}`);
          const data = await response.json();

          if (data.status === 'processing' || data.status === 'calibrating') {
            updateProgress(data);
          } else if (data.status === 'complete') {
            handleComplete(data);
            stopPolling();
            setProcessing(false);
          } else if (data.status === 'error') {
            setError(data.message || 'Processing failed');
            stopPolling();
            setProcessing(false);
          }
        } catch (err) {
          console.error('Polling error:', err);
        }
      }, 500);

      return () => stopPolling();
    }
  }, [jobId, processing]);

  const stopPolling = () => {
    if (pollingInterval.current) {
      clearInterval(pollingInterval.current);
      pollingInterval.current = null;
    }
  };

  const updateProgress = (data: any) => {
    setCurrentFrame(data.frame || 0);
    setTotalFrames(data.total_frames || 0);
    setProgress(data.progress_percent || 0);

    if (Array.isArray(data.score)) {
      setScore([data.score[0] || 0, data.score[1] || 0]);
    }
    setRallyState(data.rally_state || '');

    Animated.timing(progressAnim, {
      toValue: data.progress_percent || 0,
      duration: 300,
      useNativeDriver: false,
    }).start();
  };

  const handleComplete = (data: any) => {
    setFinalResult(data);
    if (Array.isArray(data.score)) {
      setScore([data.score[0] || 0, data.score[1] || 0]);
    }
    Alert.alert(
      'Processing Complete',
      `Final score: ${data.score?.[0] ?? 0} - ${data.score?.[1] ?? 0}\nLandings detected: ${data.total_landings ?? 0}`,
      [{ text: 'OK' }]
    );
  };

  const pickVideo = async () => {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      setError('Media library permission is required');
      return;
    }

    const res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Videos,
      quality: 1,
    });

    if (!res.canceled) {
      setVideoUri(res.assets[0].uri);
      resetState();
    }
  };

  const recordVideo = async () => {
    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (!permission.granted) {
      setError('Camera permission is required');
      return;
    }

    const res = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Videos,
      quality: 1,
    });

    if (!res.canceled) {
      setVideoUri(res.assets[0].uri);
      resetState();
    }
  };

  const resetState = () => {
    stopPolling();
    setJobId(null);
    setFinalResult(null);
    setError(null);
    setCurrentFrame(0);
    setTotalFrames(0);
    setProgress(0);
    setScore([0, 0]);
    setRallyState('');
    progressAnim.setValue(0);
  };

  const chooseVideoSource = () => {
    Alert.alert('Select Video Source', '', [
      { text: 'Camera', onPress: recordVideo },
      { text: 'Gallery', onPress: pickVideo },
      { text: 'Cancel', style: 'cancel' },
    ]);
  };

  const uploadAndProcessVideo = async () => {
    if (!videoUri || processing) return;

    setProcessing(true);
    resetState();

    const formData = new FormData();
    formData.append('file', {
      uri: videoUri,
      name: 'badminton_video.mp4',
      type: 'video/mp4',
    } as any);

    try {
      const uploadResponse = await fetch(`${API_BASE}/upload-video`, {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (!uploadResponse.ok) throw new Error('Upload failed');

      const uploadData = await uploadResponse.json();
      if (!uploadData.job_id) throw new Error('No job ID received');

      const processResponse = await fetch(`${API_BASE}/process-video/${uploadData.job_id}`, {
        method: 'POST',
      });

      if (!processResponse.ok) throw new Error('Could not start processing');

      setJobId(uploadData.job_id);
    } catch (err) {
      setError('Upload failed. Please check server or network.');
      console.error(err);
      setProcessing(false);
    }
  };

  const jobActive = processing || finalResult !== null;

  return (
    <View style={styles.root}>
      <ShuttleBackground>
      <SafeAreaView style={styles.safe} edges={['top', 'bottom']}>
        <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>

          {/* Header */}
          <View style={styles.header}>
            <Text style={styles.title}>Game Tracker</Text>
          </View>

          {/* Action card */}
          <View style={styles.glassCard}>
            <View style={styles.tileRow}>
              <TouchableOpacity
                style={[styles.tile, videoUri && !processing ? styles.tileSelected : null]}
                onPress={chooseVideoSource}
                activeOpacity={0.8}
                disabled={processing}
              >
                <Ionicons name="cloud-upload-outline" size={26} color={colors.accent} />
                <Text style={styles.tileLabel}>Analyze video</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[styles.tile, styles.tileLive]}
                onPress={() => router.push('/live')}
                activeOpacity={0.8}
                disabled={processing}
              >
                <Ionicons name="radio-outline" size={26} color={colors.accent} />
                <Text style={[styles.tileLabel, { color: colors.accent }]}>Go live</Text>
              </TouchableOpacity>
            </View>
          </View>

          {/* Video preview */}
          {videoUri && !processing && (
            <TouchableOpacity
              style={styles.glassCard}
              onPress={chooseVideoSource}
              activeOpacity={0.85}
            >
              <Video
                source={{ uri: videoUri }}
                style={styles.video}
                useNativeControls
                isLooping
                resizeMode={ResizeMode.CONTAIN}
              />
            </TouchableOpacity>
          )}

          {/* Start analysis button */}
          {videoUri && !processing && !finalResult && (
            <TouchableOpacity
              style={styles.ctaButton}
              onPress={uploadAndProcessVideo}
              activeOpacity={0.85}
            >
              <Ionicons name="play" size={20} color={colors.ctaText} />
              <Text style={styles.ctaText}>Start Analysis</Text>
            </TouchableOpacity>
          )}

          {/* Status card */}
          {jobActive && (
            <View style={styles.glassCard}>
              <View style={styles.scoreRow}>
                <View style={styles.scoreChip}>
                  <Text style={styles.scoreDigit}>{score[0]}</Text>
                </View>
                <View style={styles.scoreChip}>
                  <Text style={styles.scoreDigit}>{score[1]}</Text>
                </View>
              </View>

              {processing && (
                <>
                  <View style={styles.progressTrack}>
                    <Animated.View
                      style={[
                        styles.progressFill,
                        {
                          width: progressAnim.interpolate({
                            inputRange: [0, 100],
                            outputRange: ['0%', '100%'],
                          }),
                        },
                      ]}
                    />
                  </View>
                  <Text style={styles.monoLabel}>
                    {rallyState
                      ? rallyState === 'RALLY_ACTIVE'
                        ? 'RALLY IN PLAY'
                        : 'WAITING FOR SERVE'
                      : `${currentFrame} / ${totalFrames} FRAMES · ${progress.toFixed(0)}%`}
                  </Text>
                  <ActivityIndicator color={colors.accent} size="small" style={{ marginTop: 8 }} />
                </>
              )}

              {finalResult && (
                <>
                  <Text style={styles.monoLabel}>
                    {finalResult.total_landings ?? 0} LANDINGS DETECTED
                  </Text>
                  <Video
                    source={{ uri: `${API_BASE}/download/${jobId}` }}
                    style={styles.resultsVideo}
                    useNativeControls
                    isLooping
                    resizeMode={ResizeMode.CONTAIN}
                  />
                </>
              )}
            </View>
          )}

          {/* Error */}
          {error && (
            <View style={styles.errorCard}>
              <Ionicons name="warning-outline" size={20} color={colors.destructive} />
              <Text style={styles.errorText}>{error}</Text>
            </View>
          )}

        </ScrollView>
      </SafeAreaView>
      </ShuttleBackground>
    </View>
  );
}

const GLASS = {
  backgroundColor: colors.surface,
  borderWidth: 1,
  borderColor: colors.surfaceBorder,
  borderRadius: 18,
};

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: colors.background },
  safe: { flex: 1 },
  scroll: { padding: spacing(2.5), paddingBottom: spacing(5) },

  header: { marginBottom: spacing(3) },
  title: {
    fontFamily: fonts.heading,
    fontSize: 32,
    color: colors.textPrimary,
    letterSpacing: 0.5,
  },

  glassCard: {
    ...GLASS,
    marginBottom: spacing(2),
    overflow: 'hidden',
  },
  tileRow: { flexDirection: 'row', gap: 1 },
  tile: {
    flex: 1,
    paddingVertical: spacing(2.5),
    paddingHorizontal: spacing(2),
    alignItems: 'center',
    gap: spacing(1),
  },
  tileSelected: { backgroundColor: colors.accentDim },
  tileLive: {
    backgroundColor: colors.accentDim,
    borderLeftWidth: 1,
    borderLeftColor: colors.surfaceBorder,
  },
  tileLabel: {
    fontFamily: fonts.bodyMedium,
    fontSize: 13,
    color: colors.textPrimary,
    textAlign: 'center',
  },

  video: { width: '100%', height: 220 },

  ctaButton: {
    flexDirection: 'row',
    height: 56,
    borderRadius: 14,
    backgroundColor: colors.ctaBackground,
    borderWidth: 1,
    borderColor: colors.ctaBorder,
    justifyContent: 'center',
    alignItems: 'center',
    gap: spacing(1),
    marginBottom: spacing(2),
  },
  ctaText: {
    fontFamily: fonts.bodyBold,
    fontSize: 17,
    color: colors.ctaText,
  },

  scoreRow: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: spacing(1.5),
    paddingTop: spacing(2),
    paddingHorizontal: spacing(2),
  },
  scoreChip: {
    backgroundColor: colors.accentDim,
    borderWidth: 1,
    borderColor: colors.accentBorder,
    borderRadius: 14,
    paddingHorizontal: spacing(3),
    paddingVertical: spacing(0.5),
  },
  scoreDigit: {
    fontFamily: fonts.heading,
    fontSize: 36,
    color: colors.accent,
    fontVariant: ['tabular-nums'],
  },

  progressTrack: {
    height: 4,
    backgroundColor: colors.accentDim,
    borderRadius: 2,
    overflow: 'hidden',
    marginHorizontal: spacing(2),
    marginTop: spacing(1.5),
  },
  progressFill: {
    height: '100%',
    backgroundColor: colors.accent,
    borderRadius: 2,
  },

  monoLabel: {
    fontFamily: fonts.mono,
    fontSize: 10,
    color: colors.textMuted,
    letterSpacing: 2,
    textAlign: 'center',
    marginVertical: spacing(1),
    paddingHorizontal: spacing(2),
  },

  resultsVideo: {
    width: '100%',
    height: 260,
    borderRadius: 12,
    marginTop: spacing(1),
  },

  errorCard: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing(1.5),
    backgroundColor: 'rgba(220,38,38,0.12)',
    borderWidth: 1,
    borderColor: colors.destructive,
    borderRadius: 12,
    padding: spacing(2),
    marginBottom: spacing(2),
  },
  errorText: {
    fontFamily: fonts.body,
    fontSize: 14,
    color: colors.destructive,
    flex: 1,
  },
});
