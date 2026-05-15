#!/usr/bin/env node

import pkgJson from '../package.json' with { type: 'json' };

const args = process.argv.slice(2);
const command = args[0];
const subcommand = args[1];

function printHelp() {
  console.log(`
mlx - MLX-Node CLI v${pkgJson.version}

Usage:
  mlx <command> [options]

Commands:
  download model       Download a model from HuggingFace
  download dataset     Download a dataset from HuggingFace
  convert              Convert model weights to MLX format
  convert lora-fuse    Fuse a privacy-filter LoRA adapter into base weights
  redact               Redact PII from text using a privacy-filter model
  train privacy-lora   LoRA fine-tune a privacy-filter checkpoint
  launch claude        Start a local server and spawn Claude Code pointed at it

Options:
  -h, --help           Show this help message
  -v, --version        Show version number

Examples:
  mlx download model -m Qwen/Qwen3-0.6B
  mlx download dataset -d openai/gsm8k
  mlx convert -i ~/.mlx-node/models/qwen3-0.6b -o ~/.mlx-node/models/qwen3-0.6b-mlx -d bf16
  mlx convert lora-fuse --adapter ./out/run-001 --out ./out/run-001-fused
  mlx redact -m .cache/models/privacy-filter -i input.txt -o redacted.txt
  mlx train privacy-lora --config train.json
  mlx launch claude
`);
}

async function main() {
  if (!command || command === '--help' || command === '-h') {
    printHelp();
    return;
  }

  if (command === '--version' || command === '-v') {
    console.log(pkgJson.version);
    return;
  }

  switch (command) {
    case 'download': {
      if (!subcommand || subcommand === '--help' || subcommand === '-h') {
        console.log(`
Usage:
  mlx download model     Download a model from HuggingFace
  mlx download dataset   Download a dataset from HuggingFace

Run mlx download <subcommand> --help for more information.
`);
        return;
      }

      const rest = args.slice(2);

      if (subcommand === 'model') {
        const { run } = await import('./commands/download-model.js');
        await run(rest);
      } else if (subcommand === 'dataset') {
        const { run } = await import('./commands/download-dataset.js');
        await run(rest);
      } else {
        console.error(`Unknown download subcommand: ${subcommand}`);
        console.error('Available: model, dataset');
        process.exit(1);
      }
      break;
    }

    case 'convert': {
      // Route `mlx convert lora-fuse ...` to the LoRA-fuse subcommand.
      // Plain `mlx convert -i ... -o ...` (no subcommand) falls through to
      // the legacy weight converter for backwards compatibility.
      if (subcommand === 'lora-fuse') {
        const rest = args.slice(2);
        const { run } = await import('./commands/convert-lora.js');
        await run(rest);
      } else {
        const rest = args.slice(1);
        const { run } = await import('./commands/convert.js');
        await run(rest);
      }
      break;
    }

    case 'redact': {
      const rest = args.slice(1);
      const { run } = await import('./commands/redact.js');
      await run(rest);
      break;
    }

    case 'train': {
      if (!subcommand || subcommand === '--help' || subcommand === '-h') {
        console.log(`
Usage:
  mlx train privacy-lora     LoRA fine-tune a privacy-filter checkpoint

Run mlx train <subcommand> --help for more information.
`);
        return;
      }
      const rest = args.slice(2);
      if (subcommand === 'privacy-lora') {
        const { run } = await import('./commands/train-privacy.js');
        await run(rest);
      } else {
        console.error(`Unknown train subcommand: ${subcommand}`);
        console.error('Available: privacy-lora');
        process.exit(1);
      }
      break;
    }

    case 'launch': {
      const rest = args.slice(2);
      if (!subcommand || subcommand === '--help' || subcommand === '-h') {
        console.log(`
Usage:
  mlx launch claude     Start a local server and spawn Claude Code pointed at it

Run mlx launch <subcommand> --help for more information.
`);
        return;
      }
      if (subcommand === 'claude') {
        const { run } = await import('./commands/launch-claude/index.js');
        await run(rest);
      } else {
        console.error(`Unknown launch subcommand: ${subcommand}`);
        console.error('Available: claude');
        process.exit(1);
      }
      break;
    }

    default:
      console.error(`Unknown command: ${command}`);
      printHelp();
      process.exit(1);
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
