import { makeAutoObservable } from 'mobx';
import AsyncStorage from '@react-native-async-storage/async-storage';

/**
 * Settings Store - MobX state management for app settings
 * 
 * This store manages user preferences including the Broadcast API toggle
 */

interface SettingsStoreData {
  // Existing settings would go here
  broadcastApiEnabled: boolean;
  broadcastApiPort: number;
}

class SettingsStore {
  private readonly STORAGE_KEY = '@pocketpal_settings';
  
  // State
  broadcastApiEnabled = false;
  broadcastApiPort = 11434;
  isLoading = false;
  error: string | null = null;

  constructor() {
    makeAutoObservable(this);
    this.loadSettings();
  }

  /**
   * Load settings from AsyncStorage
   */
  async loadSettings(): Promise<void> {
    try {
      this.isLoading = true;
      this.error = null;

      const stored = await AsyncStorage.getItem(this.STORAGE_KEY);
      if (stored) {
        const data = JSON.parse(stored) as Partial<SettingsStoreData>;
        this.broadcastApiEnabled = data.broadcastApiEnabled ?? false;
        this.broadcastApiPort = data.broadcastApiPort ?? 11434;
      }
    } catch (error) {
      this.error = error instanceof Error ? error.message : 'Failed to load settings';
      console.error('Error loading settings:', error);
    } finally {
      this.isLoading = false;
    }
  }

  /**
   * Save settings to AsyncStorage
   */
  private async saveSettings(): Promise<void> {
    try {
      const data: SettingsStoreData = {
        broadcastApiEnabled: this.broadcastApiEnabled,
        broadcastApiPort: this.broadcastApiPort,
      };
      await AsyncStorage.setItem(this.STORAGE_KEY, JSON.stringify(data));
    } catch (error) {
      this.error = error instanceof Error ? error.message : 'Failed to save settings';
      console.error('Error saving settings:', error);
    }
  }

  /**
   * Toggle Broadcast API
   */
  async setBroadcastApiEnabled(enabled: boolean): Promise<void> {
    this.broadcastApiEnabled = enabled;
    await this.saveSettings();
  }

  /**
   * Set Broadcast API port
   */
  async setBroadcastApiPort(port: number): Promise<void> {
    if (port < 1024 || port > 65535) {
      throw new Error('Port must be between 1024 and 65535');
    }
    this.broadcastApiPort = port;
    await this.saveSettings();
  }

  /**
   * Get current settings
   */
  getSettings(): SettingsStoreData {
    return {
      broadcastApiEnabled: this.broadcastApiEnabled,
      broadcastApiPort: this.broadcastApiPort,
    };
  }

  /**
   * Clear all settings
   */
  async clearSettings(): Promise<void> {
    try {
      await AsyncStorage.removeItem(this.STORAGE_KEY);
      this.broadcastApiEnabled = false;
      this.broadcastApiPort = 11434;
    } catch (error) {
      this.error = error instanceof Error ? error.message : 'Failed to clear settings';
      console.error('Error clearing settings:', error);
    }
  }
}

// Export singleton instance
export const settingsStore = new SettingsStore();
