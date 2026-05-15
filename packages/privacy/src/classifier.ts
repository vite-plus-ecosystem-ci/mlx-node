import { PrivacyFilterModel as NativePrivacyFilterModel } from '@mlx-node/core';

import { redactImpl } from './redactor.js';
import type { ClassifyOptions, ClassifyResult, Entity, RedactOptions, RedactResult } from './types.js';

/**
 * High-level wrapper around the native `PrivacyFilterModel` NAPI class.
 *
 * The native binding's `load`/`classify` methods are synchronous, but the
 * public API here is intentionally `async` so we can later move work to a
 * worker thread without a breaking change to consumers.
 */
export class PrivacyFilter {
  private constructor(private readonly native: NativePrivacyFilterModel) {}

  /**
   * Load a privacy-filter checkpoint from a directory.
   *
   * The directory must contain `config.json`, `model.safetensors`,
   * `tokenizer.json`, and optionally `viterbi_calibration.json` and
   * `tokenizer_config.json`.
   */
  static async load(modelPath: string): Promise<PrivacyFilter> {
    // Native `load` is synchronous; we keep the public surface async so
    // future implementations (e.g. off-main-thread load) don't break ABI.
    const native = NativePrivacyFilterModel.load(modelPath);
    return new PrivacyFilter(native);
  }

  /**
   * Classify `text` and return detected PII entities (and optionally
   * per-token tags when `opts.returnTokens` is `true`).
   */
  async classify(text: string, opts?: ClassifyOptions): Promise<ClassifyResult> {
    const result = this.native.classify(text, opts ?? null);
    // The native binding returns `label: string`; at runtime any
    // non-background span is one of the 8 `PrivacyLabel` values, so the
    // cast through `Entity[]` narrows the public surface for callers.
    return {
      entities: result.entities as Entity[],
      tokens: result.tokens ?? undefined,
    };
  }

  /**
   * Classify `text`, then replace each detected entity according to
   * `opts.replacement` (defaults to `[<label>]`). If `opts.labels` is set,
   * only entities whose label is in that list are redacted.
   */
  async redact(text: string, opts?: RedactOptions): Promise<RedactResult> {
    const { entities } = await this.classify(text, opts);
    return redactImpl(text, entities, opts);
  }

  /**
   * Attach a LoRA adapter from a directory containing
   * `adapter.safetensors` and `adapter_config.json`. The adapter is
   * applied to every attention projection (q/k/v/o) and overrides the
   * classifier head's `score.weight` / `score.bias`. Any previously
   * loaded adapter is cleared first.
   *
   * `adapter_config.json` must provide `rank` (number), `alpha` (number),
   * and `dropout` (number). `target_modules` and `base_model_path` are
   * informational and may be absent.
   */
  async loadAdapter(adapterDir: string): Promise<void> {
    // Native `loadAdapter` is synchronous; we keep the public surface
    // async to mirror the rest of the wrapper and leave room for future
    // off-main-thread loading without an ABI break.
    this.native.loadAdapter(adapterDir);
  }

  /**
   * Drop all LoRA adapters and restore the original classifier head.
   * Idempotent — calling on a model without an adapter is a no-op.
   */
  async clearAdapter(): Promise<void> {
    this.native.clearAdapter();
  }
}
