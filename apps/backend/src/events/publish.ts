import { EventEmitter } from 'node:events';

export interface BacktestCompletedEvent {
  jobId: string;
  profileId: string;
  metrics: Record<string, unknown>;
  evidenceId: string;
  traceId: string;
}

type BacktestEventListener = (event: BacktestCompletedEvent) => void | Promise<void>;

const emitter = new EventEmitter();
const EVENT_NAME = 'naxs.backtest.completed.v1';

export async function publishBacktestCompleted(event: BacktestCompletedEvent): Promise<void> {
  const listeners = emitter.listeners(EVENT_NAME);

  for (const listener of listeners) {
    await (listener as BacktestEventListener)(event);
  }
}

export function onBacktestCompleted(listener: BacktestEventListener): () => void {
  const wrapper = (event: BacktestCompletedEvent): void => {
    void listener(event);
  };

  emitter.on(EVENT_NAME, wrapper);

  return () => {
    emitter.removeListener(EVENT_NAME, wrapper);
  };
}

export function resetEventListeners(): void {
  emitter.removeAllListeners(EVENT_NAME);
}
