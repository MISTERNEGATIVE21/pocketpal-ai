import { LlamaContext } from 'llama.rn';

/**
 * Llama API Handler
 * 
 * Bridges OpenAI-compatible API requests to llama.rn LlamaContext.
 * Handles streaming and non-streaming responses.
 */

interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

interface ChatCompletionRequest {
  messages: ChatMessage[];
  model: string;
  temperature?: number;
  max_tokens?: number;
  stream?: boolean;
  top_p?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
}

interface ChatCompletionResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: string;
      content: string;
    };
    finish_reason: string;
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

interface ChatCompletionStreamResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    delta: {
      role?: string;
      content?: string;
    };
    finish_reason: string | null;
  }>;
}

class LlamaApiHandler {
  private llamaContext: LlamaContext | null = null;

  /**
   * Initialize with a LlamaContext instance
   */
  setLlamaContext(context: LlamaContext): void {
    this.llamaContext = context;
  }

  /**
   * Handle chat completion requests (OpenAI-compatible)
   */
  async handleChatCompletion(request: ChatCompletionRequest): Promise<string> {
    if (!this.llamaContext) {
      throw new Error('LlamaContext not initialized');
    }

    const {
      messages,
      model,
      temperature = 0.7,
      max_tokens = 512,
      stream = false,
      top_p = 0.9,
    } = request;

    // Format messages for llama.rn
    const formattedMessages = messages
      .map(msg => `${msg.role}: ${msg.content}`)
      .join('\n');

    try {
      if (stream) {
        return this.handleStreamingCompletion(
          formattedMessages,
          model,
          temperature,
          max_tokens,
          top_p
        );
      } else {
        return this.handleNonStreamingCompletion(
          formattedMessages,
          model,
          temperature,
          max_tokens,
          top_p
        );
      }
    } catch (error) {
      console.error('Error handling chat completion:', error);
      throw error;
    }
  }

  /**
   * Handle streaming completions
   */
  private async handleStreamingCompletion(
    prompt: string,
    model: string,
    temperature: number,
    maxTokens: number,
    topP: number
  ): Promise<string> {
    if (!this.llamaContext) {
      throw new Error('LlamaContext not initialized');
    }

    let streamContent = '';
    const completionId = `chatcmpl-${Date.now()}`;
    const createdAt = Math.floor(Date.now() / 1000);

    // Stream the response
    try {
      // This is a placeholder - actual streaming implementation depends on llama.rn API
      const completion = await this.llamaContext.complete(prompt, {
        temperature,
        top_p: topP,
        n_predict: maxTokens,
      });

      streamContent = completion;
    } catch (error) {
      console.error('Error in streaming completion:', error);
      throw error;
    }

    // Format as server-sent events (SSE) for streaming
    const lines: string[] = [];

    // Split response into tokens and send as stream
    const tokens = streamContent.split(' ');
    for (const token of tokens) {
      const response: ChatCompletionStreamResponse = {
        id: completionId,
        object: 'text_completion.chunk',
        created: createdAt,
        model: model,
        choices: [
          {
            index: 0,
            delta: { content: token + ' ' },
            finish_reason: null,
          },
        ],
      };
      lines.push(`data: ${JSON.stringify(response)}`);
    }

    // Send final message
    const finalResponse: ChatCompletionStreamResponse = {
      id: completionId,
      object: 'text_completion.chunk',
      created: createdAt,
      model: model,
      choices: [
        {
          index: 0,
          delta: { content: '' },
          finish_reason: 'stop',
        },
      ],
    };
    lines.push(`data: ${JSON.stringify(finalResponse)}`);
    lines.push('data: [DONE]');

    return lines.join('\n');
  }

  /**
   * Handle non-streaming completions
   */
  private async handleNonStreamingCompletion(
    prompt: string,
    model: string,
    temperature: number,
    maxTokens: number,
    topP: number
  ): Promise<string> {
    if (!this.llamaContext) {
      throw new Error('LlamaContext not initialized');
    }

    try {
      const completion = await this.llamaContext.complete(prompt, {
        temperature,
        top_p: topP,
        n_predict: maxTokens,
      });

      const response: ChatCompletionResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'text_completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: completion,
            },
            finish_reason: 'stop',
          },
        ],
        usage: {
          prompt_tokens: prompt.split(' ').length,
          completion_tokens: completion.split(' ').length,
          total_tokens:
            prompt.split(' ').length + completion.split(' ').length,
        },
      };

      return JSON.stringify(response);
    } catch (error) {
      console.error('Error in non-streaming completion:', error);
      throw error;
    }
  }
}

export const llamaApiHandler = new LlamaApiHandler();
