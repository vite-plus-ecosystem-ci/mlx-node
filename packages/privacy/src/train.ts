import {
  PrivacyLoraTrainerJs as NativePrivacyLoraTrainerJs,
  type PrivacyLoraTrainConfigJs as NativePrivacyLoraTrainConfigJs,
} from '@mlx-node/core';

/**
 * Configuration for {@link PrivacyLoraTrainer.create}.
 *
 * All tunables are optional; omitting a field falls back to the
 * documented defaults from the native trainer:
 *
 * - `rank` = 16, `alpha` = 32, `dropout` = 0.05
 * - `loraLr` = 1e-4, `classifierLr` = 5e-5
 * - `batchSize` = 2, `maxSeqLen` = 256, `numEpochs` = 3
 * - `gradAccumSteps` = 4, `gradClip` = 1.0
 * - `saveEvery` = 500 (set 0 to disable intermediate checkpoints)
 * - `padTokenId` = 0
 *
 * `modelPath`, `dataPath`, and `outputDir` are required.
 *
 * The field names match the native NAPI casing
 * ({@link import('@mlx-node/core').PrivacyLoraTrainConfigJs}) so the
 * wrapper can pass values straight through without re-keying.
 */
export interface PrivacyLoraTrainConfig {
  /** Filesystem path to the base privacy-filter checkpoint directory. */
  modelPath: string;
  /** Path to the training JSONL dataset (pre-tokenized). */
  dataPath: string;
  /**
   * Optional evaluation JSONL dataset path. Currently ignored by the
   * trainer — reserved for future eval-loop integration.
   */
  evalPath?: string;
  /**
   * Directory where `adapter.safetensors`, `adapter_config.json`,
   * `optimizer_state.safetensors`, and `trainer_state.json` are written.
   */
  outputDir: string;
  rank?: number;
  alpha?: number;
  dropout?: number;
  loraLr?: number;
  classifierLr?: number;
  batchSize?: number;
  maxSeqLen?: number;
  numEpochs?: number;
  gradAccumSteps?: number;
  gradClip?: number;
  saveEvery?: number;
  resumeFrom?: string;
  padTokenId?: number;
}

/**
 * High-level wrapper around the native `PrivacyLoraTrainerJs` NAPI class.
 *
 * The native binding's `create` factory is synchronous (it loads the base
 * checkpoint, attaches zero-B LoRA adapters, and parses the JSONL dataset
 * inline), but the public API here is intentionally `async` so we can later
 * move work to a worker thread without a breaking change to consumers.
 *
 * `train()` and `saveAdapter()` are already async on the native side — they
 * funnel through `spawn_blocking` so the Node.js event loop stays responsive
 * for the duration of the training run.
 */
export class PrivacyLoraTrainer {
  private constructor(private readonly inner: NativePrivacyLoraTrainerJs) {}

  /**
   * Construct a new trainer.
   *
   * Loads the base privacy-filter model from `config.modelPath`, attaches
   * zero-B LoRA adapters on every attention projection, and parses the
   * pre-tokenized JSONL dataset at `config.dataPath`.
   */
  static async create(config: PrivacyLoraTrainConfig): Promise<PrivacyLoraTrainer> {
    // NAPI requires the camelCase keys to exactly match the interface — pass
    // the object through directly so any future field additions are picked
    // up without an extra mapping table.
    const native = NativePrivacyLoraTrainerJs.create(config satisfies NativePrivacyLoraTrainConfigJs);
    return new PrivacyLoraTrainer(native);
  }

  /**
   * Run the full training loop. Resolves with the final optimizer step
   * count once `numEpochs` epochs have completed.
   */
  async train(): Promise<number> {
    return this.inner.train();
  }

  /**
   * Persist adapter weights, optimizer state, and trainer progress to the
   * configured `outputDir`. Overwrites any previous checkpoint at that
   * path.
   */
  async saveAdapter(): Promise<void> {
    return this.inner.saveAdapter();
  }
}
