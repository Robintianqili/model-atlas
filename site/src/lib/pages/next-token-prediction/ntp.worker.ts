/// <reference lib="webworker" />
/**
 * Web Worker for the next-token-prediction page: every TinyGPT computation (scoring, next-token
 * logits, generation, checkpoint loading) runs here so the page stays responsive.
 * The protocol is in engine.ts (createHost).
 */
import { createHost, type Reply, type Request } from './engine.ts';

const scope = self as unknown as DedicatedWorkerGlobalScope;
const handle = createHost((r: Reply) => scope.postMessage(r), fetch.bind(scope));
scope.onmessage = (e: MessageEvent<Request>) => {
	void handle(e.data);
};
