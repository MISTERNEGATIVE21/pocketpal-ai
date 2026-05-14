import { makeAutoObservable } from 'mobx';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { v4 as uuidv4 } from 'uuid';

/**
 * API Keys Store - MobX state management for Broadcast API keys
 * 
 * Manages API key generation, storage, and access control for the Broadcast API
 */

export interface ApiKeyData {
  id: string;
  key: string;
  name: string;
  createdAt: number;
  lastUsed?: number;
  isActive: boolean;
}

class ApiKeysStore {
  private readonly STORAGE_KEY = '@pocketpal_api_keys';

  // State
  apiKeys: ApiKeyData[] = [];
  isLoading = false;
  error: string | null = null;

  constructor() {
    makeAutoObservable(this);
    this.loadApiKeys();
  }

  /**
   * Load API keys from AsyncStorage
   */
  async loadApiKeys(): Promise<void> {
    try {
      this.isLoading = true;
      this.error = null;

      const stored = await AsyncStorage.getItem(this.STORAGE_KEY);
      if (stored) {
        this.apiKeys = JSON.parse(stored);
      }
    } catch (error) {
      this.error = error instanceof Error ? error.message : 'Failed to load API keys';
      console.error('Error loading API keys:', error);
    } finally {
      this.isLoading = false;
    }
  }

  /**
   * Save API keys to AsyncStorage
   */
  private async saveApiKeys(): Promise<void> {
    try {
      await AsyncStorage.setItem(this.STORAGE_KEY, JSON.stringify(this.apiKeys));
    } catch (error) {
      this.error = error instanceof Error ? error.message : 'Failed to save API keys';
      console.error('Error saving API keys:', error);
    }
  }

  /**
   * Generate a new API key
   */
  async generateApiKey(name: string): Promise<ApiKeyData> {
    try {
      const newKey: ApiKeyData = {
        id: uuidv4(),
        key: this.generateSecureKey(),
        name: name || `Key ${new Date().toLocaleDateString()}`,
        createdAt: Date.now(),
        isActive: true,
      };

      this.apiKeys.push(newKey);
      await this.saveApiKeys();

      return newKey;
    } catch (error) {
      this.error = error instanceof Error ? error.message : 'Failed to generate API key';
      throw error;
    }
  }

  /**
   * Generate a secure random API key
   */
  private generateSecureKey(): string {
    // Format: pk_<random_string>
    const randomPart = uuidv4().replace(/-/g, '');
    return `pk_${randomPart}`;
  }

  /**
   * Delete an API key
   */
  async deleteApiKey(keyId: string): Promise<void> {
    try {
      this.apiKeys = this.apiKeys.filter(k => k.id !== keyId);
      await this.saveApiKeys();
    } catch (error) {
      this.error = error instanceof Error ? error.message : 'Failed to delete API key';
      throw error;
    }
  }

  /**
   * Toggle API key active status
   */
  async toggleApiKeyStatus(keyId: string, isActive: boolean): Promise<void> {
    try {
      const keyIndex = this.apiKeys.findIndex(k => k.id === keyId);
      if (keyIndex !== -1) {
        this.apiKeys[keyIndex].isActive = isActive;
        await this.saveApiKeys();
      }
    } catch (error) {
      this.error = error instanceof Error ? error.message : 'Failed to update API key';
      throw error;
    }
  }

  /**
   * Rename an API key
   */
  async renameApiKey(keyId: string, newName: string): Promise<void> {
    try {
      const keyIndex = this.apiKeys.findIndex(k => k.id === keyId);
      if (keyIndex !== -1) {
        this.apiKeys[keyIndex].name = newName;
        await this.saveApiKeys();
      }
    } catch (error) {
      this.error = error instanceof Error ? error.message : 'Failed to rename API key';
      throw error;
    }
  }

  /**
   * Validate an API key (check if it's valid and active)
   */
  validateApiKey(key: string): ApiKeyData | null {
    const foundKey = this.apiKeys.find(k => k.key === key && k.isActive);
    if (foundKey) {
      // Update last used timestamp
      foundKey.lastUsed = Date.now();
      this.saveApiKeys().catch(err => 
        console.error('Failed to update last used timestamp:', err)
      );
      return foundKey;
    }
    return null;
  }

  /**
   * Get all active API keys
   */
  getActiveApiKeys(): ApiKeyData[] {
    return this.apiKeys.filter(k => k.isActive);
  }

  /**
   * Get all API keys
   */
  getAllApiKeys(): ApiKeyData[] {
    return this.apiKeys;
  }

  /**
   * Clear all API keys
   */
  async clearAllApiKeys(): Promise<void> {
    try {
      await AsyncStorage.removeItem(this.STORAGE_KEY);
      this.apiKeys = [];
    } catch (error) {
      this.error = error instanceof Error ? error.message : 'Failed to clear API keys';
      console.error('Error clearing API keys:', error);
    }
  }
}

// Export singleton instance
export const apiKeysStore = new ApiKeysStore();
