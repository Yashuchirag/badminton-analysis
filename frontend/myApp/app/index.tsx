import { useState } from 'react';
import {
  StyleSheet,
  ActivityIndicator,
  TouchableOpacity,
  View,
  ScrollView,
  Alert
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { Image } from 'expo-image';

import { ThemedView } from '@/components/themed-view';
import { ThemedText } from '@/components/themed-text';

export default function HomeScreen() {
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any | null>(null);
  const [error, setError] = useState<string | null>(null);

  /* Pick image from gallery */
  const pickImage = async () => {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      setError('Media library permission is required');
      return;
    }

    const res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 1,
    });

    if (!res.canceled) {
      setImageUri(res.assets[0].uri);
      setResult(null);
      setError(null);
    }
  };

  /* Take Photo */
  const takePhoto = async () => {
    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (!permission.granted) {
      setError('Camera permission is required');
      return;
    }

    const res = await ImagePicker.launchCameraAsync({
      quality: 1,
    });

    if (!res.canceled) {
      setImageUri(res.assets[0].uri);
      setResult(null);
      setError(null);
    }
  };

  /* ------------------ Source Selector ------------------ */
  const chooseImageSource = () => {
    Alert.alert('Select Image Source', '', [
      { text: 'Camera', onPress: takePhoto },
      { text: 'Gallery', onPress: pickImage },
      { text: 'Cancel', style: 'cancel' },
    ]);
  };

  /* Upload Image */
  const uploadImage = async () => {
    if (!imageUri || loading) return;

    setLoading(true);
    setResult(null);
    setError(null);

    const formData = new FormData();
    formData.append('file', {
      uri: imageUri,
      name: 'image.jpg',
      type: 'image/jpeg',
    } as any);

    try {
      const response = await fetch('http://192.168.68.55:8000/track-human', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Server error');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError('Upload failed. Please check server or network.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.scroll}>
      <ThemedView style={styles.container}>
        <ThemedText type="title" style={styles.title}>
          Human Tracking
        </ThemedText>

        <ThemedText style={styles.subtitle}>
          Capture or Select an image to detect humans
        </ThemedText>

        {/* Image Selector */}
        <TouchableOpacity
          style={styles.imageCard}
          onPress={chooseImageSource}
          activeOpacity={0.85}
        >
          {imageUri ? (
            <Image
              source={{ uri: imageUri }}
              style={styles.image}
              contentFit="cover"
            />
          ) : (
            <ThemedText style={styles.placeholderText}>
              Tap to choose image
            </ThemedText>
          )}
        </TouchableOpacity>

        {/* Upload Button */}
        <TouchableOpacity
          style={[
            styles.uploadButton,
            (!imageUri || loading) && styles.disabledButton,
          ]}
          onPress={uploadImage}
          disabled={!imageUri || loading}
        >
          {loading ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <ThemedText style={styles.uploadText}>
              Upload
            </ThemedText>
          )}
        </TouchableOpacity>

        {/* Error */}
        {error && (
          <ThemedText style={styles.errorText}>{error}</ThemedText>
        )}

        {/* Result Card */}
        {result && (
          <View style={styles.resultCard}>
            <ThemedText type="subtitle" style={styles.resultTitle}>
              Detection Results
            </ThemedText>

            {/* Example structured fields (safe if undefined) */}
            {result.humans_detected !== undefined && (
              <ThemedText>
                Humans detected:{' '}
                <ThemedText style={styles.bold}>
                  {result.humans_detected}
                </ThemedText>
              </ThemedText>
            )}

            {result.processing_time && (
              <ThemedText>
                Processing time: {result.processing_time} ms
              </ThemedText>
            )}

            {/* Raw JSON fallback */}
            <View style={styles.jsonBox}>
              <ThemedText style={styles.jsonText}>
                {JSON.stringify(result, null, 2)}
              </ThemedText>
            </View>
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
    justifyContent: 'center',
  },
  title: {
    textAlign: 'center',
    marginBottom: 4,
  },
  subtitle: {
    textAlign: 'center',
    opacity: 0.7,
    marginBottom: 20,
  },
  imageCard: {
    height: 260,
    borderRadius: 16,
    backgroundColor: '#E5E7EB',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 20,
    overflow: 'hidden',
  },
  image: {
    width: '100%',
    height: '100%',
  },
  placeholderText: {
    opacity: 0.6,
  },
  uploadButton: {
    height: 52,
    borderRadius: 14,
    backgroundColor: '#2563EB',
    justifyContent: 'center',
    alignItems: 'center',
  },
  disabledButton: {
    backgroundColor: '#9CA3AF',
  },
  uploadText: {
    color: '#fff',
    fontWeight: '600',
    fontSize: 16,
  },
  errorText: {
    marginTop: 12,
    color: '#DC2626',
    textAlign: 'center',
  },
  resultCard: {
    marginTop: 24,
    padding: 18,
    borderRadius: 16,
    backgroundColor: '#0F172A', // slate-900 (very dark)
    borderWidth: 1,
    borderColor: '#334155', // slate-700
    shadowColor: '#000',
    shadowOpacity: 0.3,
    shadowRadius: 14,
    shadowOffset: { width: 0, height: 6 },
    elevation: 8,
  },

  resultTitle: {
    marginBottom: 10,
    fontWeight: '700',
    fontSize: 16,
    color: '#F8FAFC', // near-white
  },

  bold: {
    fontWeight: '700',
    color: '#38BDF8', // cyan accent
  },

  jsonBox: {
    marginTop: 12,
    padding: 12,
    borderRadius: 12,
    backgroundColor: '#020617', // even darker
    borderWidth: 1,
    borderColor: '#1E293B',
  },

  jsonText: {
    fontSize: 12,
    color: '#E5E7EB',
  },

});
