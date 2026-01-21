import { useState } from 'react';
import {
  StyleSheet,
  ActivityIndicator,
  TouchableOpacity,
  View,
  ScrollView,
  Alert,
  Text,
  Dimensions
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { Video, ResizeMode } from 'expo-av';

import { ThemedView } from '@/components/themed-view';
import { ThemedText } from '@/components/themed-text';

const { width } = Dimensions.get('window');

interface TrackingResult {
  status: string;
  device_used: string;
  total_frames: number;
  unique_ids: number[];
  total_unique_people: number;
  wrist_data_points: number;
  wrist_samples: Array<{
    frame: number;
    left_wrist: { x: number; y: number };
    right_wrist: { x: number; y: number };
  }>;
  shuttle_detections: number;
  shuttle_positions: Array<[number, number] | null>;
  output_video?: string;
}

export default function HomeScreen() {
  const [videoUri, setVideoUri] = useState<string | null>(null);
  const [annotatedVideoUri, setAnnotatedVideoUri] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<TrackingResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);


  /* Pick video from gallery */
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
      setAnnotatedVideoUri(null);
      setResult(null);
      setError(null);
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
      setAnnotatedVideoUri(null);
      setResult(null);
      setError(null);
    }
  };


  const chooseVideoSource = () => {
    Alert.alert('Select Video Source', '', [
      { text: 'Camera', onPress: recordVideo },
      { text: 'Gallery', onPress: pickVideo },
      { text: 'Cancel', style: 'cancel' },
    ]);
  };

  /* Upload video */
  const uploadVideo = async () => {
    if (!videoUri || loading) return;

    setLoading(true);
    setResult(null);
    setError(null);
    setUploadProgress(0);

    const formData = new FormData();
    formData.append('file', {
      uri: videoUri,
      name: 'badminton_video.mp4',
      type: 'video/mp4',
    } as any);

    try {
      const response = await fetch('http://192.168.68.71:8000/track-human-video', {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (!response.ok) throw new Error('Server error');

      const data: TrackingResult = await response.json();
      setResult(data);
      
      // If output video path is provided, download it
      if (data.output_video) {
        downloadAnnotatedVideo(data.output_video);
      }
      
      console.log('Tracking complete:', data);
    } catch (err) {
      setError('Upload failed. Please check server or network.');
    } finally {
      setLoading(false);
    }
  };

  const downloadAnnotatedVideo = async (videoPath: string) => {
    try {
      const filename = videoPath.split('/').pop() || 'annotated.mp4';
      const response = await fetch(
        `http://192.168.68.71:8000/download-video/${encodeURIComponent(videoPath)}`
      );
      
      if (response.ok) {
        const blob = await response.blob();
        
        setAnnotatedVideoUri(videoPath);
      }
    } catch (err) {
      console.error('Failed to download annotated video:', err);
    }
  };

  const renderStatCard = (label: string, value: string | number, color: string) => (
    <View style={[styles.statCard, { borderLeftColor: color }]}>
      <ThemedText style={styles.statLabel}>{label}</ThemedText>
      <ThemedText style={styles.statValue}>{value}</ThemedText>
    </View>
  );
  
  return (
    <ScrollView contentContainerStyle={styles.scroll}>
      <ThemedView style={styles.container}>
        <ThemedText type="title" style={styles.title}>
          🏸 Badminton Tracker
        </ThemedText>

        <ThemedText style={styles.subtitle}>
          Track players, wrists, and shuttle in real-time
        </ThemedText>

        {/* Video Selector */}
        <TouchableOpacity
          style={styles.videoCard}
          onPress={chooseVideoSource}
          activeOpacity={0.85}
        >
          {videoUri ? (
            <Video
              source={{ uri: videoUri }}
              style={styles.video}
              useNativeControls
              isLooping
              resizeMode={ResizeMode.CONTAIN}
            />
          ) : (
            <View style={styles.placeholder}>
              <Text style={styles.placeholderIcon}>📹</Text>
              <ThemedText style={styles.placeholderText}>
                Tap to select badminton video
              </ThemedText>
            </View>
          )}
        </TouchableOpacity>

        {/* Upload Button */}
        <TouchableOpacity
          style={[styles.uploadButton, (!videoUri || loading) && styles.disabledButton]}
          onPress={uploadVideo}
          disabled={!videoUri || loading}
        >
          {loading ? (
            <View style={styles.loadingContainer}>
              <ActivityIndicator color="#fff" size="small" />
              <ThemedText style={styles.uploadText}>
                Processing... This may take a while
              </ThemedText>
            </View>
          ) : (
            <ThemedText style={styles.uploadText}>
              🚀 Analyze Video
            </ThemedText>
          )}
        </TouchableOpacity>


        {/* Error */}
        {error && (
          <View style={styles.errorCard}>
            <Text style={styles.errorIcon}>⚠️</Text>
            <ThemedText style={styles.errorText}>{error}</ThemedText>
          </View>
        )}

        {/* Result Card */}
        {result && (
          <View style={styles.resultsContainer}>
            <ThemedText type="subtitle" style={styles.sectionTitle}>
              📊 Analysis Results
            </ThemedText>

            {/* Quick Stats Grid */}
            <View style={styles.statsGrid}>
              {renderStatCard('Total Frames', result.total_frames, '#3B82F6')}
              {renderStatCard('Players Detected', result.total_unique_people, '#10B981')}
              {renderStatCard('Shuttle Detections', result.shuttle_detections, '#F59E0B')}
              {renderStatCard('Wrist Keypoints', result.wrist_data_points, '#8B5CF6')}
            </View>

            {/* Device Info */}
            <View style={styles.infoCard}>
              <Text style={styles.infoIcon}>🖥️</Text>
              <ThemedText style={styles.infoText}>
                Processed on: <ThemedText style={styles.bold}>{result.device_used.toUpperCase()}</ThemedText>
              </ThemedText>
            </View>

            {/* Player IDs */}
            {result.unique_ids && result.unique_ids.length > 0 && (
              <View style={styles.detailCard}>
                <ThemedText style={styles.detailTitle}>👥 Player IDs</ThemedText>
                <View style={styles.badgeContainer}>
                  {result.unique_ids.map((id) => (
                    <View key={id} style={styles.badge}>
                      <ThemedText style={styles.badgeText}>#{id}</ThemedText>
                    </View>
                  ))}
                </View>
              </View>
            )}

            {/* Wrist Sample Data */}
            {result.wrist_samples && result.wrist_samples.length > 0 && (
              <View style={styles.detailCard}>
                <ThemedText style={styles.detailTitle}>🤚 Wrist Tracking Sample</ThemedText>
                <ThemedText style={styles.sampleNote}>
                  Showing first {result.wrist_samples.length} detections
                </ThemedText>
                {result.wrist_samples.slice(0, 3).map((sample, idx) => (
                  <View key={idx} style={styles.sampleRow}>
                    <ThemedText style={styles.sampleFrame}>Frame {sample.frame}</ThemedText>
                    <View style={styles.wristCoords}>
                      <View style={styles.coordBox}>
                        <ThemedText style={styles.coordLabel}>L</ThemedText>
                        <ThemedText style={styles.coordValue}>
                          ({Math.round(sample.left_wrist.x)}, {Math.round(sample.left_wrist.y)})
                        </ThemedText>
                      </View>
                      <View style={styles.coordBox}>
                        <ThemedText style={styles.coordLabel}>R</ThemedText>
                        <ThemedText style={styles.coordValue}>
                          ({Math.round(sample.right_wrist.x)}, {Math.round(sample.right_wrist.y)})
                        </ThemedText>
                      </View>
                    </View>
                  </View>
                ))}
              </View>
            )}

            {/* Shuttle Detection Rate */}
            <View style={styles.detailCard}>
              <ThemedText style={styles.detailTitle}>🏸 Shuttle Detection Rate</ThemedText>
              <View style={styles.progressBar}>
                <View 
                  style={[
                    styles.progressFill, 
                    { 
                      width: `${(result.shuttle_detections / result.total_frames) * 100}%` 
                    }
                  ]} 
                />
              </View>
              <ThemedText style={styles.percentageText}>
                {((result.shuttle_detections / result.total_frames) * 100).toFixed(1)}% 
                ({result.shuttle_detections}/{result.total_frames} frames)
              </ThemedText>
            </View>

            {/* Download Annotated Video Button */}
            {annotatedVideoUri && (
              <TouchableOpacity style={styles.downloadButton}>
                <Text style={styles.downloadIcon}>📥</Text>
                <ThemedText style={styles.downloadText}>
                  Annotated Video Ready
                </ThemedText>
              </TouchableOpacity>
            )}

            {/* Raw JSON (Collapsible) */}
            <TouchableOpacity 
              style={styles.jsonToggle}
              onPress={() => Alert.alert('Raw Data', JSON.stringify(result, null, 2))}
            >
              <ThemedText style={styles.jsonToggleText}>
                📋 View Raw JSON Data
              </ThemedText>
            </TouchableOpacity>
          </View>
        )}
      </ThemedView>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scroll: {
    flexGrow: 1,
  },
  container: {
    flex: 1,
    padding: 20,
  },
  title: {
    textAlign: 'center',
    marginBottom: 4,
    fontSize: 28,
  },
  subtitle: {
    textAlign: 'center',
    opacity: 0.7,
    marginBottom: 24,
    fontSize: 14,
  },
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
  video: {
    width: '100%',
    height: '100%',
  },
  placeholder: {
    alignItems: 'center',
  },
  placeholderIcon: {
    fontSize: 48,
    marginBottom: 12,
  },
  placeholderText: {
    opacity: 0.6,
    fontSize: 14,
  },
  uploadButton: {
    height: 56,
    borderRadius: 14,
    backgroundColor: '#2563EB',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 16,
    shadowColor: '#2563EB',
    shadowOpacity: 0.3,
    shadowRadius: 8,
    shadowOffset: { width: 0, height: 4 },
    elevation: 5,
  },
  disabledButton: {
    backgroundColor: '#64748B',
    shadowOpacity: 0,
  },
  loadingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  uploadText: {
    color: '#fff',
    fontWeight: '700',
    fontSize: 16,
  },
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
  errorIcon: {
    fontSize: 24,
  },
  errorText: {
    color: '#FCA5A5',
    flex: 1,
  },
  resultsContainer: {
    marginTop: 8,
  },
  sectionTitle: {
    marginBottom: 16,
    fontWeight: '700',
    fontSize: 20,
  },
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
  statLabel: {
    fontSize: 12,
    opacity: 0.7,
    marginBottom: 4,
  },
  statValue: {
    fontSize: 24,
    fontWeight: '700',
    color: '#38BDF8',
  },
  infoCard: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    borderRadius: 12,
    backgroundColor: '#0F172A',
    borderWidth: 1,
    borderColor: '#334155',
    marginBottom: 16,
    gap: 12,
  },
  infoIcon: {
    fontSize: 24,
  },
  infoText: {
    fontSize: 14,
    opacity: 0.9,
  },
  bold: {
    fontWeight: '700',
    color: '#38BDF8',
  },
  detailCard: {
    padding: 16,
    borderRadius: 12,
    backgroundColor: '#0F172A',
    borderWidth: 1,
    borderColor: '#334155',
    marginBottom: 16,
  },
  detailTitle: {
    fontSize: 16,
    fontWeight: '700',
    marginBottom: 12,
    color: '#F8FAFC',
  },
  badgeContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  badge: {
    backgroundColor: '#1E40AF',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
  },
  badgeText: {
    color: '#DBEAFE',
    fontSize: 12,
    fontWeight: '600',
  },
  sampleNote: {
    fontSize: 12,
    opacity: 0.6,
    marginBottom: 12,
  },
  sampleRow: {
    marginBottom: 12,
    paddingBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#1E293B',
  },
  sampleFrame: {
    fontSize: 12,
    fontWeight: '600',
    marginBottom: 8,
    color: '#94A3B8',
  },
  wristCoords: {
    flexDirection: 'row',
    gap: 12,
  },
  coordBox: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  coordLabel: {
    fontSize: 10,
    fontWeight: '700',
    backgroundColor: '#1E40AF',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
    color: '#DBEAFE',
  },
  coordValue: {
    fontSize: 11,
    fontFamily: 'monospace',
    opacity: 0.8,
  },
  progressBar: {
    height: 12,
    backgroundColor: '#1E293B',
    borderRadius: 6,
    overflow: 'hidden',
    marginBottom: 8,
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#F59E0B',
    borderRadius: 6,
  },
  percentageText: {
    fontSize: 12,
    textAlign: 'center',
    opacity: 0.8,
  },
  downloadButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
    borderRadius: 12,
    backgroundColor: '#065F46',
    borderWidth: 1,
    borderColor: '#10B981',
    marginBottom: 16,
    gap: 12,
  },
  downloadIcon: {
    fontSize: 24,
  },
  downloadText: {
    color: '#D1FAE5',
    fontWeight: '600',
    fontSize: 14,
  },
  jsonToggle: {
    padding: 14,
    borderRadius: 12,
    backgroundColor: '#0F172A',
    borderWidth: 1,
    borderColor: '#334155',
    alignItems: 'center',
  },
  jsonToggleText: {
    fontSize: 13,
    opacity: 0.7,
  },
});