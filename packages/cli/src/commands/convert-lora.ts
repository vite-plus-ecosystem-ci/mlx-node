import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';

import { PrivacyFilterModel } from '@mlx-node/core';

function printHelp(): void {
  console.log(`
Fuse a privacy-filter LoRA adapter into the base weights

Usage:
  mlx convert lora-fuse --adapter <dir> --out <dir> [options]

Required Arguments:
  --adapter <dir>           Output directory produced by 'mlx train privacy-lora'
                            (contains adapter.safetensors + adapter_config.json).
  --out <dir>               Directory to write the fused model.safetensors.

Optional Arguments:
  --base <path>             Override the base-model path. By default the
                            base path is read from adapter_config.json's
                            "base_model_path" field.
  -h, --help                Show this help message

Notes:
  - Errors if any LoRA adapter sits on top of a quantized projection.
    Re-fuse with the dense base checkpoint instead.
  - The in-memory model state is the fused-weight view; the original base
    checkpoint on disk is unchanged.

Examples:
  mlx convert lora-fuse --adapter ./out/run-001 --out ./out/run-001-fused
  mlx convert lora-fuse --adapter ./out/run-001 --out ./fused --base .cache/models/privacy-filter
`);
}

export async function run(argv: string[]): Promise<void> {
  const { values: args } = parseArgs({
    args: argv,
    options: {
      adapter: { type: 'string' },
      out: { type: 'string' },
      base: { type: 'string' },
      help: { type: 'boolean', short: 'h', default: false },
    },
  });

  if (args.help) {
    printHelp();
    return;
  }

  if (!args.adapter) {
    console.error('Error: --adapter is required');
    process.exit(1);
  }
  if (!args.out) {
    console.error('Error: --out is required');
    process.exit(1);
  }

  const adapterDir = resolve(args.adapter);
  const outDir = resolve(args.out);

  if (!existsSync(adapterDir)) {
    console.error(`Error: adapter directory not found: ${adapterDir}`);
    process.exit(1);
  }

  const adapterConfigPath = resolve(adapterDir, 'adapter_config.json');
  if (!existsSync(adapterConfigPath)) {
    console.error(`Error: missing adapter_config.json in ${adapterDir}`);
    process.exit(1);
  }

  // Resolve the base model path from either --base or adapter_config.json's
  // "base_model_path" field (written by the trainer at save time).
  let basePath: string | undefined = args.base ? resolve(args.base) : undefined;
  if (!basePath) {
    try {
      const cfg = JSON.parse(readFileSync(adapterConfigPath, 'utf-8')) as {
        base_model_path?: string;
      };
      if (typeof cfg.base_model_path === 'string' && cfg.base_model_path.length > 0) {
        basePath = resolve(cfg.base_model_path);
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`Error: failed to parse ${adapterConfigPath}: ${msg}`);
      process.exit(1);
    }
  }
  if (!basePath) {
    console.error(
      'Error: could not resolve base model path. Pass --base or ensure adapter_config.json has "base_model_path".',
    );
    process.exit(1);
  }
  if (!existsSync(basePath)) {
    console.error(`Error: base model directory not found: ${basePath}`);
    process.exit(1);
  }

  console.log('Privacy LoRA fuse');
  console.log(`  Base model:  ${basePath}`);
  console.log(`  Adapter:     ${adapterDir}`);
  console.log(`  Output:      ${outDir}`);
  console.log('');

  const startTime = Date.now();
  let model: PrivacyFilterModel;
  try {
    model = PrivacyFilterModel.load(basePath);
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`Error: failed to load base model: ${msg}`);
    process.exit(1);
  }

  try {
    model.loadAdapter(adapterDir);
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`Error: failed to load adapter: ${msg}`);
    process.exit(1);
  }

  try {
    model.fuseAdapter(outDir);
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`Error: fuse failed: ${msg}`);
    process.exit(1);
  }

  const duration = ((Date.now() - startTime) / 1000).toFixed(2);
  console.log(`Fused model written to ${outDir}`);
  console.log(`Duration: ${duration}s`);
}
