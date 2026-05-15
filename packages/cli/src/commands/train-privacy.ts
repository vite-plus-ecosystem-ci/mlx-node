import { readFileSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';

import { PrivacyLoraTrainer, type PrivacyLoraTrainConfig } from '@mlx-node/privacy';

function printHelp(): void {
  console.log(`
LoRA fine-tune the privacy-filter token-classification model

Usage:
  mlx train privacy-lora [--config <file.json>] [options]

Configuration:
  Pass --config <file.json> to load all settings from a JSON file, or pass
  the individual flags below. CLI flags always win over file values when
  both are present.

Required Arguments (either via --config or via flags):
  --model <path>            Base privacy-filter checkpoint directory.
                            Must contain config.json, model.safetensors,
                            tokenizer.json, viterbi_calibration.json.
  --data <path>             Pre-tokenized training JSONL dataset.
  --output <dir>            Directory for adapter.safetensors,
                            adapter_config.json, optimizer_state.safetensors,
                            and trainer_state.json.

Optional Arguments:
  --config <file>           JSON file with any of the keys below in camelCase
                            (modelPath, dataPath, outputDir, rank, alpha,
                             dropout, loraLr, classifierLr, batchSize,
                             maxSeqLen, numEpochs, gradAccumSteps, gradClip,
                             saveEvery, resumeFrom, padTokenId, evalPath).
  --eval <path>             Evaluation JSONL (reserved; not yet used).
  --rank <int>              LoRA rank (default: 16).
  --alpha <float>           LoRA alpha (default: 32).
  --dropout <float>         LoRA dropout (default: 0.05).
  --lora-lr <float>         AdamW LR for LoRA A/B (default: 1e-4).
  --classifier-lr <float>   AdamW LR for classifier head (default: 5e-5).
  --batch-size <int>        Examples per micro-batch (default: 2).
  --max-seq-len <int>       Max sequence length (default: 256).
  --num-epochs <int>        Number of epochs (default: 3).
  --grad-accum-steps <int>  Micro-batches per optimizer step (default: 4).
  --grad-clip <float>       L2 gradient clip norm (default: 1.0).
  --save-every <int>        Save every N steps; 0 disables (default: 500).
  --resume-from <dir>       Resume from a previous output directory.
  --pad-token-id <int>      Token id used for padding (default: 0).
  -h, --help                Show this help message

Examples:
  mlx train privacy-lora --config train.json
  mlx train privacy-lora \\
    --model .cache/models/privacy-filter \\
    --data ./data/train.jsonl \\
    --output ./out/run-001 \\
    --num-epochs 5 --batch-size 4
`);
}

/** Parse a non-negative integer flag (zero is accepted); exits on invalid input. */
function parseNonNegativeInt(flag: string, raw?: string): number | undefined {
  if (raw === undefined) return undefined;
  if (!/^[0-9]+$/.test(raw)) {
    console.error(`Error: ${flag} requires a non-negative integer value`);
    process.exit(1);
  }
  return Number(raw);
}

/** Parse a finite float flag; exits on invalid input. */
function parseFloatFlag(flag: string, raw?: string): number | undefined {
  if (raw === undefined) return undefined;
  const v = Number(raw);
  if (!Number.isFinite(v)) {
    console.error(`Error: ${flag} requires a finite numeric value`);
    process.exit(1);
  }
  return v;
}

export async function run(argv: string[]): Promise<void> {
  const { values: args } = parseArgs({
    args: argv,
    options: {
      config: { type: 'string' },
      model: { type: 'string' },
      data: { type: 'string' },
      output: { type: 'string' },
      eval: { type: 'string' },
      rank: { type: 'string' },
      alpha: { type: 'string' },
      dropout: { type: 'string' },
      'lora-lr': { type: 'string' },
      'classifier-lr': { type: 'string' },
      'batch-size': { type: 'string' },
      'max-seq-len': { type: 'string' },
      'num-epochs': { type: 'string' },
      'grad-accum-steps': { type: 'string' },
      'grad-clip': { type: 'string' },
      'save-every': { type: 'string' },
      'resume-from': { type: 'string' },
      'pad-token-id': { type: 'string' },
      help: { type: 'boolean', short: 'h', default: false },
    },
  });

  if (args.help) {
    printHelp();
    return;
  }

  // 1. Start with any keys provided by --config (file values).
  // 2. Overlay CLI flags so they take precedence.
  let fileConfig: Partial<PrivacyLoraTrainConfig> = {};
  if (args.config) {
    const configPath = resolve(args.config);
    if (!existsSync(configPath)) {
      console.error(`Error: config file not found: ${configPath}`);
      process.exit(1);
    }
    try {
      fileConfig = JSON.parse(readFileSync(configPath, 'utf-8')) as Partial<PrivacyLoraTrainConfig>;
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`Error: failed to parse config file ${configPath}: ${msg}`);
      process.exit(1);
    }
  }

  const merged: Partial<PrivacyLoraTrainConfig> = {
    ...fileConfig,
    ...(args.model !== undefined ? { modelPath: resolve(args.model) } : {}),
    ...(args.data !== undefined ? { dataPath: resolve(args.data) } : {}),
    ...(args.output !== undefined ? { outputDir: resolve(args.output) } : {}),
    ...(args.eval !== undefined ? { evalPath: resolve(args.eval) } : {}),
    ...(args.rank !== undefined ? { rank: parseNonNegativeInt('--rank', args.rank) } : {}),
    ...(args.alpha !== undefined ? { alpha: parseFloatFlag('--alpha', args.alpha) } : {}),
    ...(args.dropout !== undefined ? { dropout: parseFloatFlag('--dropout', args.dropout) } : {}),
    ...(args['lora-lr'] !== undefined ? { loraLr: parseFloatFlag('--lora-lr', args['lora-lr']) } : {}),
    ...(args['classifier-lr'] !== undefined
      ? { classifierLr: parseFloatFlag('--classifier-lr', args['classifier-lr']) }
      : {}),
    ...(args['batch-size'] !== undefined ? { batchSize: parseNonNegativeInt('--batch-size', args['batch-size']) } : {}),
    ...(args['max-seq-len'] !== undefined ? { maxSeqLen: parseNonNegativeInt('--max-seq-len', args['max-seq-len']) } : {}),
    ...(args['num-epochs'] !== undefined ? { numEpochs: parseNonNegativeInt('--num-epochs', args['num-epochs']) } : {}),
    ...(args['grad-accum-steps'] !== undefined
      ? { gradAccumSteps: parseNonNegativeInt('--grad-accum-steps', args['grad-accum-steps']) }
      : {}),
    ...(args['grad-clip'] !== undefined ? { gradClip: parseFloatFlag('--grad-clip', args['grad-clip']) } : {}),
    ...(args['save-every'] !== undefined ? { saveEvery: parseNonNegativeInt('--save-every', args['save-every']) } : {}),
    ...(args['resume-from'] !== undefined ? { resumeFrom: resolve(args['resume-from']) } : {}),
    ...(args['pad-token-id'] !== undefined
      ? { padTokenId: parseNonNegativeInt('--pad-token-id', args['pad-token-id']) }
      : {}),
  };

  // Required-field validation.
  if (!merged.modelPath) {
    console.error('Error: --model (or modelPath in --config) is required');
    process.exit(1);
  }
  if (!merged.dataPath) {
    console.error('Error: --data (or dataPath in --config) is required');
    process.exit(1);
  }
  if (!merged.outputDir) {
    console.error('Error: --output (or outputDir in --config) is required');
    process.exit(1);
  }

  if (!existsSync(merged.modelPath)) {
    console.error(`Error: model directory not found: ${merged.modelPath}`);
    process.exit(1);
  }
  if (!existsSync(merged.dataPath)) {
    console.error(`Error: data file not found: ${merged.dataPath}`);
    process.exit(1);
  }

  const config = merged as PrivacyLoraTrainConfig;

  console.log('Privacy LoRA Trainer');
  console.log(`  Base model:  ${config.modelPath}`);
  console.log(`  Train data:  ${config.dataPath}`);
  console.log(`  Output dir:  ${config.outputDir}`);
  if (config.rank !== undefined) console.log(`  Rank:        ${config.rank}`);
  if (config.alpha !== undefined) console.log(`  Alpha:       ${config.alpha}`);
  if (config.batchSize !== undefined) console.log(`  Batch size:  ${config.batchSize}`);
  if (config.maxSeqLen !== undefined) console.log(`  Max seq len: ${config.maxSeqLen}`);
  if (config.numEpochs !== undefined) console.log(`  Num epochs:  ${config.numEpochs}`);
  if (config.gradAccumSteps !== undefined) console.log(`  Grad accum:  ${config.gradAccumSteps}`);
  if (config.loraLr !== undefined) console.log(`  LoRA LR:     ${config.loraLr}`);
  if (config.classifierLr !== undefined) console.log(`  Cls LR:      ${config.classifierLr}`);
  if (config.resumeFrom !== undefined) console.log(`  Resume from: ${config.resumeFrom}`);
  console.log('');

  const startTime = Date.now();
  let trainer: PrivacyLoraTrainer;
  try {
    trainer = await PrivacyLoraTrainer.create(config);
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`Error: trainer creation failed: ${msg}`);
    process.exit(1);
  }

  console.log('Starting training...');
  let finalStep: number;
  try {
    finalStep = await trainer.train();
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`\nError: training failed: ${msg}`);
    process.exit(1);
  }

  console.log('\nSaving final adapter checkpoint...');
  try {
    await trainer.saveAdapter();
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`Error: saveAdapter failed: ${msg}`);
    process.exit(1);
  }

  const duration = ((Date.now() - startTime) / 1000).toFixed(2);
  console.log(`\nTraining complete.`);
  console.log(`  Final step: ${finalStep}`);
  console.log(`  Output:     ${config.outputDir}`);
  console.log(`  Duration:   ${duration}s`);
}
