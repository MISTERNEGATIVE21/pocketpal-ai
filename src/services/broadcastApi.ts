import { NativeModules, NativeEventEmitter, AppState } from 'react-native';

/**
 * Broadcast API Service
 * 
 * This service manages the local HTTP API server that listens on port 11434
 * and accepts OpenAI-compatible POST requests to /v1/chat/completions.
 * 
 * It routes requests to the active LlamaContext from llama.rn and returns
 * stream or JSON responses through the HTTP bridge.
 */

const { HttpBridge } = NativeModules;

interface BroadcastApiRequest {
  messages: Array<{ role: string; content: string }>;
  model: string;
  temperature?: number;
  max_tokens?: number;
  stream?: boolean;
}

interface BroadcastApiResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message?: { role: string; content: string };
    delta?: { content: string };
    finish_reason: string | null;
  }>;
}

class BroadcastApiService {
  private isActive = false;
  private httpBridgeEventEmitter: NativeEventEmitter | null = null;
  private appStateSubscription: any = null;
  private requestHandler: ((request: BroadcastApiRequest) => Promise<string>) | null = null;

  constructor() {
    this.setupEventListeners();
  }

  /**
   * Initialize the Broadcast API service
   */
  private setupEventListeners() {
    if (HttpBridge && typeof HttpBridge.addListener === 'function') {
      this.httpBridgeEventEmitter = new NativeEventEmitter(HttpBridge);
      
      // Listen for incoming HTTP requests
      this.httpBridgeEventEmitter.addListener(
        'onHttpRequest',
        this.handleHttpRequest.bind(this)
      );
    }
  }

  /**
   * Start the Broadcast API server
   */
  async startServer(
    requestHandler: (request: BroadcastApiRequest) => Promise<string>
  ): Promise<void> {
    if (this.isActive) {
      console.warn('Broadcast API is already running');
      return;
    }

    try {
      this.requestHandler = requestHandler;
      
      // Start the HTTP server on port 11434
      if (HttpBridge && typeof HttpBridge.startServer === 'function') {
        await HttpBridge.startServer(11434);
        this.isActive = true;
        console.log('Broadcast API server started on port 11434');
      }

      // Start foreground service to keep the API alive
      this.startForegroundService();

      // Listen for app state changes
      this.appStateSubscription = AppState.addEventListener('change', this.handleAppStateChange.bind(this));
    } catch (error) {
      console.error('Failed to start Broadcast API server:', error);
      throw error;
    }
  }

  /**
   * Stop the Broadcast API server
   */
  async stopServer(): Promise<void> {
    if (!this.isActive) {
      console.warn('Broadcast API is not running');
      return;
    }

    try {
      if (HttpBridge && typeof HttpBridge.stopServer === 'function') {
        await HttpBridge.stopServer();
        this.isActive = false;
        console.log('Broadcast API server stopped');
      }

      this.stopForegroundService();
      this.requestHandler = null;

      if (this.appStateSubscription) {
        this.appStateSubscription.remove();
        this.appStateSubscription = null;
      }
    } catch (error) {
      console.error('Failed to stop Broadcast API server:', error);
      throw error;
    }
  }

  /**
   * Handle incoming HTTP requests
   */
  private async handleHttpRequest(event: {
    requestId: string;
    path: string;
    method: string;
    body: string;
  }): Promise<void> {
    try {
      const { requestId, path, method, body } = event;

      // Only handle POST requests to /v1/chat/completions
      if (method !== 'POST' || path !== '/v1/chat/completions') {
        this.sendHttpResponse(requestId, 404, { error: 'Not found' });
        return;
      }

      const request = JSON.parse(body) as BroadcastApiRequest;

      if (!this.requestHandler) {
        this.sendHttpResponse(requestId, 500, { error: 'Request handler not configured' });
        return;
      }

      // Process the request through llama.rn
      const response = await this.requestHandler(request);

      // Send the response back
      this.sendHttpResponse(requestId, 200, response, request.stream);
    } catch (error) {
      console.error('Error handling HTTP request:', error);
      this.sendHttpResponse(event.requestId, 500, {
        error: error instanceof Error ? error.message : 'Internal server error',
      });
    }
  }

  /**
   * Send HTTP response back to the client
   */
  private sendHttpResponse(
    requestId: string,
    statusCode: number,
    body: any,
    isStream: boolean = false
  ): void {
    try {
      if (HttpBridge && typeof HttpBridge.sendResponse === 'function') {
        const responseBody = typeof body === 'string' ? body : JSON.stringify(body);
        const headers = {
          'Content-Type': isStream ? 'text/event-stream' : 'application/json',
          'Access-Control-Allow-Origin': '*',
        };

        HttpBridge.sendResponse(requestId, statusCode, responseBody, headers);
      }
    } catch (error) {
      console.error('Error sending HTTP response:', error);
    }
  }

  /**
   * Start the foreground service to keep the API alive
   */
  private startForegroundService(): void {
    try {
      const { ForegroundService } = NativeModules;
      if (ForegroundService && typeof ForegroundService.startService === 'function') {
        ForegroundService.startService({
          notificationTitle: 'PocketPal API',
          notificationText: 'Local API server is running on port 11434',
          notificationId: 9999,
        });
        console.log('Foreground service started');
      }
    } catch (error) {
      console.error('Failed to start foreground service:', error);
    }
  }

  /**
   * Stop the foreground service
   */
  private stopForegroundService(): void {
    try {
      const { ForegroundService } = NativeModules;
      if (ForegroundService && typeof ForegroundService.stopService === 'function') {
        ForegroundService.stopService();
        console.log('Foreground service stopped');
      }
    } catch (error) {
      console.error('Failed to stop foreground service:', error);
    }
  }

  /**
   * Handle app state changes (foreground/background)
   */
  private handleAppStateChange(nextAppState: string): void {
    if (nextAppState === 'background' && this.isActive) {
      console.log('App moved to background, ensuring foreground service is active');
      // Ensure foreground service is running
      this.startForegroundService();
    }
  }

  /**
   * Check if the API server is active
   */
  isServerActive(): boolean {
    return this.isActive;
  }
}

export const broadcastApiService = new BroadcastApiService();
