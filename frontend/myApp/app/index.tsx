import { useState, useRef, useEffect } from 'react';
import {
  StyleSheet,
  ActivityIndicator,
  TouchableOpacity,
  View,
  ScrollView,
  Alert,
  Text,
  Dimensions,
  Animated
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { Image } from 'expo-image';
import { Video, ResizeMode } from 'expo-av';

import { ThemedView } from '@/components/themed-view';
import { ThemedText } from '@/components/themed-text';

const { width } = Dimensions.get('window');

interface StreamUpdate {
  type: 'started' | 'progress' | 'complete' | 'error';
  frame?: number;
  total_frames?: number;
  progress_percent?: number;
  people_count?: number;
  unique_people?: number;
  shuttle_detected?: boolean;
  shuttle_position?: [number, number] | null;
  preview_image?: string;
  output_video?: string;
  message?: string;
}

export default function HomeScreen() {
  const [videoUri, setVideoUri] = useState<string | null>(null);
  const [processing, setProcessing] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [totalFrames, setTotalFrames] = useState(0);
  const [progress, setProgress] = useState(0);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [liveStats, setLiveStats] = useState({
    people: 0,
    uniquePeople: 0,
    shuttleDetected: false
  });
  const [finalResult, setFinalResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  
  const progressAnim = useRef(new Animated.Value(0)).current;
  const pollingInterval = useRef<number | null>(null);

  useEffect(() => {
    if (jobId && processing) {
      pollingInterval.current = setInterval(async () => {
        try {
          const response = await fetch(`http://192.168.68.70:8000/job-status/${jobId}`);
          const data = await response.json();
          
          if (data.status === 'processing') {
            updateProgress(data);
          } else if (data.status === 'complete') {
            handleComplete(data);
            stopPolling();
          } else if (data.status === 'error') {
            setError(data.message || 'Processing failed');
            stopPolling();
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
    
    if (data.preview_image) {
      setPreviewImage(data.preview_image);
    }
    
    setLiveStats({
      people: data.people_count || 0,
      uniquePeople: data.unique_people || 0,
      shuttleDetected: data.shuttle_detected || false
    });

    Animated.timing(progressAnim, {
      toValue: data.progress_percent || 0,
      duration: 300,
      useNativeDriver: false
    }).start();
  };

  const handleComplete = (data: any) => {
    setFinalResult(data);
    Alert.alert(
      'Processing Complete! 🎉',
      `Detected ${data.unique_people} players\nShuttle detected in ${data.summary?.detection_rate}% of frames`,
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
    setPreviewImage(null);
    setFinalResult(null);
    setError(null);
    setCurrentFrame(0);
    setTotalFrames(0);
    setProgress(0);
    setLiveStats({ people: 0, uniquePeople: 0, shuttleDetected: false });
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
      const response = await fetch('http://192.168.68.70:8000/track-human-video-async', {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (!response.ok) throw new Error('Server error');

      const data = await response.json();
      
      if (data.job_id) {
        setJobId(data.job_id);
        setTotalFrames(data.total_frames || 0);
      } else {
        throw new Error('No job ID received');
      }

    } catch (err) {
      setError('Upload failed. Please check server or network.');
      console.error(err);
      setProcessing(false);
    }
  };

  const downloadVideo = async () => {
    if (!finalResult?.output_video) return;

    try {
      const url = `http://192.168.68.70:8000/download-video?path=${encodeURIComponent(finalResult.output_video)}`;
      Alert.alert('Download Ready', `Video URL: ${url}`, [
        { text: 'OK' }
      ]);
      // In production, use expo-file-system to download and save
    } catch (err) {
      console.error('Download failed:', err);
    }
  };


  return (
    <ScrollView contentContainerStyle={styles.scroll}>
      <ThemedView style={styles.container}>
        <ThemedText type="title" style={styles.title}>
          🏸 Live Badminton Tracker
        </ThemedText>

        <ThemedText style={styles.subtitle}>
          Real-time player, wrist, and shuttle tracking
        </ThemedText>

        {/* Video Selector */}
        <TouchableOpacity
          style={styles.videoCard}
          onPress={chooseVideoSource}
          activeOpacity={0.85}
          disabled={processing}
        >
          {videoUri && !processing ? (
            <Video
              source={{ uri: videoUri }}
              style={styles.video}
              useNativeControls
              isLooping
              resizeMode={ResizeMode.CONTAIN}
            />
          ) : previewImage ? (
            <Image
              source={{ uri: previewImage }}
              style={styles.video}
              contentFit="contain"
            />
          ) : (
            <View style={styles.placeholder}>
              <Text style={styles.placeholderIcon}>📹</Text>
              <ThemedText style={styles.placeholderText}>
                {processing ? 'Processing...' : 'Tap to select video'}
              </ThemedText>
            </View>
          )}
        </TouchableOpacity>

        {/* Processing Status */}
        {processing && (
          <View style={styles.statusCard}>
            <View style={styles.statusHeader}>
              <ActivityIndicator color="#3B82F6" size="small" />
              <ThemedText style={styles.statusTitle}>Processing Video</ThemedText>
            </View>
            
            {/* Progress Bar */}
            <View style={styles.progressBarContainer}>
              <Animated.View 
                style={[
                  styles.progressBarFill,
                  {
                    width: progressAnim.interpolate({
                      inputRange: [0, 100],
                      outputRange: ['0%', '100%']
                    })
                  }
                ]} 
              />
            </View>
            
            <ThemedText style={styles.progressText}>
              {currentFrame} / {totalFrames} frames ({progress.toFixed(1)}%)
            </ThemedText>

            {/* Live Stats */}
            <View style={styles.liveStatsGrid}>
              <View style={styles.liveStatBox}>
                <Text style={styles.statIcon}>👥</Text>
                <ThemedText style={styles.liveStatValue}>{liveStats.people}</ThemedText>
                <ThemedText style={styles.liveStatLabel}>Current</ThemedText>
              </View>
              <View style={styles.liveStatBox}>
                <Text style={styles.statIcon}>🏃</Text>
                <ThemedText style={styles.liveStatValue}>{liveStats.uniquePeople}</ThemedText>
                <ThemedText style={styles.liveStatLabel}>Total</ThemedText>
              </View>
              <View style={styles.liveStatBox}>
                <Text style={styles.statIcon}>🏸</Text>
                <ThemedText style={[
                  styles.liveStatValue,
                  { color: liveStats.shuttleDetected ? '#10B981' : '#64748B' }
                ]}>
                  {liveStats.shuttleDetected ? '✓' : '✗'}
                </ThemedText>
                <ThemedText style={styles.liveStatLabel}>Shuttle</ThemedText>
              </View>
            </View>
          </View>
        )}

        {/* Upload Button */}
        <TouchableOpacity
          style={[styles.uploadButton, (!videoUri || processing) && styles.disabledButton]}
          onPress={uploadAndProcessVideo}
          disabled={!videoUri || processing}
        >
          <ThemedText style={styles.uploadText}>
            {processing ? '⏳ Processing...' : '🚀 Start Analysis'}
          </ThemedText>
        </TouchableOpacity>

        {/* Error */}
        {error && (
          <View style={styles.errorCard}>
            <Text style={styles.errorIcon}>⚠️</Text>
            <ThemedText style={styles.errorText}>{error}</ThemedText>
          </View>
        )}

        {/* Final Results */}
        {finalResult && (
          <View style={styles.resultsContainer}>
            <ThemedText type="subtitle" style={styles.sectionTitle}>
              📊 Final Results
            </ThemedText>

            <View style={styles.statsGrid}>
              <View style={[styles.statCard, { borderLeftColor: '#3B82F6' }]}>
                <ThemedText style={styles.statLabel}>Total Frames</ThemedText>
                <ThemedText style={styles.statValue}>{finalResult.total_frames}</ThemedText>
              </View>
              <View style={[styles.statCard, { borderLeftColor: '#10B981' }]}>
                <ThemedText style={styles.statLabel}>Players</ThemedText>
                <ThemedText style={styles.statValue}>{finalResult.unique_people}</ThemedText>
              </View>
              <View style={[styles.statCard, { borderLeftColor: '#F59E0B' }]}>
                <ThemedText style={styles.statLabel}>Shuttle Rate</ThemedText>
                <ThemedText style={styles.statValue}>
                  {finalResult.summary?.detection_rate}%
                </ThemedText>
              </View>
              <View style={[styles.statCard, { borderLeftColor: '#8B5CF6' }]}>
                <ThemedText style={styles.statLabel}>Detections</ThemedText>
                <ThemedText style={styles.statValue}>{finalResult.shuttle_detections}</ThemedText>
              </View>
            </View>

            {/* Download Button */}
            <TouchableOpacity style={styles.downloadButton} onPress={downloadVideo}>
              <Text style={styles.downloadIcon}>📥</Text>
              <ThemedText style={styles.downloadText}>
                Download Annotated Video
              </ThemedText>
            </TouchableOpacity>
          </View>
        )}
      </ThemedView>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scroll: { flexGrow: 1 },
  container: { flex: 1, padding: 20 },
  title: { textAlign: 'center', marginBottom: 4, fontSize: 28 },
  subtitle: { textAlign: 'center', opacity: 0.7, marginBottom: 24, fontSize: 14 },
  videoCard: {
    height: 260,
    borderRadius: 16,
    backgroundColor: '#1E293B',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 16,
    overflow: 'hidden',
    borderWidth: 2,
    borderColor: '#334155',
  },
  video: { width: '100%', height: '100%' },
  placeholder: { alignItems: 'center' },
  placeholderIcon: { fontSize: 48, marginBottom: 12 },
  placeholderText: { opacity: 0.6, fontSize: 14 },
  statusCard: {
    padding: 16,
    borderRadius: 12,
    backgroundColor: '#0F172A',
    borderWidth: 1,
    borderColor: '#334155',
    marginBottom: 16,
  },
  statusHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    marginBottom: 12,
  },
  statusTitle: { fontSize: 16, fontWeight: '600' },
  progressBarContainer: {
    height: 8,
    backgroundColor: '#1E293B',
    borderRadius: 4,
    overflow: 'hidden',
    marginBottom: 8,
  },
  progressBarFill: {
    height: '100%',
    backgroundColor: '#3B82F6',
    borderRadius: 4,
  },
  progressText: {
    fontSize: 12,
    textAlign: 'center',
    opacity: 0.7,
    marginBottom: 16,
  },
  liveStatsGrid: {
    flexDirection: 'row',
    gap: 12,
  },
  liveStatBox: {
    flex: 1,
    alignItems: 'center',
    padding: 12,
    backgroundColor: '#1E293B',
    borderRadius: 8,
  },
  statIcon: { fontSize: 24, marginBottom: 4 },
  liveStatValue: {
    fontSize: 20,
    fontWeight: '700',
    color: '#38BDF8',
  },
  liveStatLabel: {
    fontSize: 10,
    opacity: 0.6,
    marginTop: 2,
  },
  uploadButton: {
    height: 56,
    borderRadius: 14,
    backgroundColor: '#2563EB',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 16,
  },
  disabledButton: { backgroundColor: '#64748B' },
  uploadText: { color: '#fff', fontWeight: '700', fontSize: 16 },
  errorCard: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    borderRadius: 12,
    backgroundColor: '#7F1D1D',
    borderWidth: 1,
    borderColor: '#DC2626',
    marginBottom: 16,
    gap: 12,
  },
  errorIcon: { fontSize: 24 },
  errorText: { color: '#FCA5A5', flex: 1 },
  resultsContainer: { marginTop: 8 },
  sectionTitle: { marginBottom: 16, fontWeight: '700', fontSize: 20 },
  statsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
    marginBottom: 16,
  },
  statCard: {
    flex: 1,
    minWidth: (width - 56) / 2,
    padding: 16,
    borderRadius: 12,
    backgroundColor: '#0F172A',
    borderWidth: 1,
    borderColor: '#334155',
    borderLeftWidth: 4,
  },
  statLabel: { fontSize: 12, opacity: 0.7, marginBottom: 4 },
  statValue: { fontSize: 24, fontWeight: '700', color: '#38BDF8' },
  downloadButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
    borderRadius: 12,
    backgroundColor: '#065F46',
    borderWidth: 1,
    borderColor: '#10B981',
    gap: 12,
  },
  downloadIcon: { fontSize: 24 },
  downloadText: { color: '#D1FAE5', fontWeight: '600', fontSize: 14 },
});