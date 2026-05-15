/**
 * Privacy-filter evaluation harness.
 *
 * Reads a JSONL eval set (`{text, entities: [{start, end, label}]}` per line —
 * same shape as the trainer's `raw.jsonl` input), runs `classify()` on each
 * row, and reports per-label precision / recall / F1 plus an overall
 * micro-averaged score.
 *
 * CLI:
 *   oxnode scripts/evaluate-privacy.ts \
 *     --model <dir> --eval <file.jsonl> [--adapter <dir>] [--metric span|token] \
 *     [--threshold 0.5] [--out report.json]
 *
 * Library:
 *   import { evaluatePrivacy } from './evaluate-privacy.ts';
 *   const report = await evaluatePrivacy(modelPath, evalPath, { adapterPath, metric });
 */
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';

import { PrivacyFilter } from '@mlx-node/privacy';

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export interface EvalEntity {
  start: number;
  end: number;
  label: string;
}

export interface EvalRow {
  text: string;
  entities: EvalEntity[];
}

export interface PerLabelStats {
  truePositives: number;
  falsePositives: number;
  falseNegatives: number;
  precision: number;
  recall: number;
  f1: number;
}

export interface EvalReport {
  modelPath: string;
  adapterPath?: string;
  evalPath: string;
  numExamples: number;
  perLabel: Record<string, PerLabelStats>;
  overall: PerLabelStats;
  metric: 'span' | 'token';
}

export interface EvaluateOptions {
  adapterPath?: string;
  metric?: 'span' | 'token';
  threshold?: number;
}

/**
 * Run an evaluation pass and return a structured report.
 *
 * - **`metric: 'span'`** (default): a predicted entity is a TP iff there is
 *   a ground-truth entity with the same `(start, end, label)`. Byte-exact.
 * - **`metric: 'token'`**: every byte position in `[start, end)` of a
 *   ground-truth entity contributes one token of that label; predictions
 *   contribute the same way; TP/FP/FN are counted per byte (effectively a
 *   character-level F1, robust to boundary jitter).
 */
export async function evaluatePrivacy(
  modelPath: string,
  evalPath: string,
  options: EvaluateOptions = {},
): Promise<EvalReport> {
  const metric = options.metric ?? 'span';
  const threshold = options.threshold ?? 0.5;

  const rows = readJsonl(evalPath);

  const pf = await PrivacyFilter.load(modelPath);
  if (options.adapterPath) {
    await pf.loadAdapter(options.adapterPath);
  }

  // tp / fp / fn per label, plus the overall micro totals.
  const counts: Record<string, { tp: number; fp: number; fn: number }> = {};
  const bump = (label: string, key: 'tp' | 'fp' | 'fn'): void => {
    if (!counts[label]) counts[label] = { tp: 0, fp: 0, fn: 0 };
    counts[label][key] += 1;
  };

  for (const row of rows) {
    const predicted = (await pf.classify(row.text, { threshold })).entities.map((e) => ({
      start: e.start,
      end: e.end,
      label: e.label as string,
    }));
    if (metric === 'span') {
      scoreSpan(row.entities, predicted, bump);
    } else {
      scoreToken(row.text.length, row.entities, predicted, bump);
    }
  }

  const perLabel: Record<string, PerLabelStats> = {};
  let totalTp = 0;
  let totalFp = 0;
  let totalFn = 0;
  for (const [label, c] of Object.entries(counts)) {
    perLabel[label] = stats(c.tp, c.fp, c.fn);
    totalTp += c.tp;
    totalFp += c.fp;
    totalFn += c.fn;
  }
  const overall = stats(totalTp, totalFp, totalFn);

  return {
    modelPath,
    adapterPath: options.adapterPath,
    evalPath,
    numExamples: rows.length,
    perLabel,
    overall,
    metric,
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function readJsonl(path: string): EvalRow[] {
  const raw = readFileSync(path, 'utf8');
  const out: EvalRow[] = [];
  for (const line of raw.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const parsed = JSON.parse(trimmed) as { text?: string; entities?: EvalEntity[] };
    if (typeof parsed.text !== 'string') {
      throw new Error(`Invalid eval row (missing "text"): ${trimmed.slice(0, 80)}`);
    }
    const entities: EvalEntity[] = Array.isArray(parsed.entities)
      ? parsed.entities.map((e) => ({ start: e.start, end: e.end, label: e.label }))
      : [];
    out.push({ text: parsed.text, entities });
  }
  return out;
}

type Bump = (label: string, key: 'tp' | 'fp' | 'fn') => void;

function scoreSpan(truth: EvalEntity[], pred: EvalEntity[], bump: Bump): void {
  const key = (e: EvalEntity): string => `${e.start}|${e.end}|${e.label}`;
  const truthKeys = new Set(truth.map(key));
  const predKeys = new Set(pred.map(key));

  for (const e of pred) {
    if (truthKeys.has(key(e))) {
      bump(e.label, 'tp');
    } else {
      bump(e.label, 'fp');
    }
  }
  for (const e of truth) {
    if (!predKeys.has(key(e))) {
      bump(e.label, 'fn');
    }
  }
}

/**
 * Character-level scoring: assign every byte in `[start, end)` of a span
 * its label and count per-label byte agreement. Resolves overlap on the
 * truth side by giving precedence to the first listed span (we never get
 * overlapping ground truth in practice; if we do, an earlier label wins).
 */
function scoreToken(textLen: number, truth: EvalEntity[], pred: EvalEntity[], bump: Bump): void {
  const truthLabels = paintLabels(textLen, truth);
  const predLabels = paintLabels(textLen, pred);
  for (let i = 0; i < textLen; i++) {
    const t = truthLabels[i];
    const p = predLabels[i];
    if (t && p) {
      if (t === p) bump(t, 'tp');
      else {
        // Disagreement: counted as FP on the predicted label and FN on
        // the truth label (standard multi-class token-F1 convention).
        bump(p, 'fp');
        bump(t, 'fn');
      }
    } else if (p && !t) {
      bump(p, 'fp');
    } else if (t && !p) {
      bump(t, 'fn');
    }
  }
}

function paintLabels(len: number, spans: EvalEntity[]): (string | null)[] {
  const out: (string | null)[] = Array.from({ length: len }, () => null);
  for (const span of spans) {
    const s = Math.max(0, span.start);
    const e = Math.min(len, span.end);
    for (let i = s; i < e; i++) {
      if (out[i] === null) out[i] = span.label;
    }
  }
  return out;
}

function stats(tp: number, fp: number, fn: number): PerLabelStats {
  const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
  const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
  const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
  return { truePositives: tp, falsePositives: fp, falseNegatives: fn, precision, recall, f1 };
}

// ---------------------------------------------------------------------------
// Pretty printing
// ---------------------------------------------------------------------------

export function formatReport(report: EvalReport): string {
  const lines: string[] = [];
  lines.push(`Privacy-filter eval (${report.metric}-level F1)`);
  lines.push(`  Model:    ${report.modelPath}`);
  if (report.adapterPath) lines.push(`  Adapter:  ${report.adapterPath}`);
  lines.push(`  Eval:     ${report.evalPath}`);
  lines.push(`  Examples: ${report.numExamples}`);
  lines.push('');

  const labels = Object.keys(report.perLabel).sort();
  const header = `  ${'label'.padEnd(22)}  ${'P'.padStart(7)}  ${'R'.padStart(7)}  ${'F1'.padStart(7)}  ${'TP'.padStart(5)}  ${'FP'.padStart(5)}  ${'FN'.padStart(5)}`;
  lines.push(header);
  lines.push(`  ${'-'.repeat(header.length - 2)}`);
  for (const label of labels) {
    const s = report.perLabel[label]!;
    lines.push(formatRow(label, s));
  }
  lines.push(`  ${'-'.repeat(header.length - 2)}`);
  lines.push(formatRow('overall', report.overall));
  return lines.join('\n');
}

function formatRow(label: string, s: PerLabelStats): string {
  const pct = (x: number): string => (x * 100).toFixed(2);
  return `  ${label.padEnd(22)}  ${pct(s.precision).padStart(7)}  ${pct(s.recall).padStart(7)}  ${pct(s.f1).padStart(7)}  ${String(s.truePositives).padStart(5)}  ${String(s.falsePositives).padStart(5)}  ${String(s.falseNegatives).padStart(5)}`;
}

// ---------------------------------------------------------------------------
// CLI entry point
// ---------------------------------------------------------------------------

function printHelp(): void {
  console.log(`
Privacy-filter eval harness — span/token F1 against a JSONL ground-truth set.

Usage:
  oxnode scripts/evaluate-privacy.ts --model <dir> --eval <file.jsonl> [options]

Required:
  --model <dir>      Path to a privacy-filter checkpoint directory.
  --eval <file>      JSONL file with one {text, entities[]} object per line.

Optional:
  --adapter <dir>    LoRA adapter directory (adapter.safetensors + adapter_config.json).
  --metric <kind>    'span' (default) or 'token'.
  --threshold <f>    Min mean per-token probability (default: 0.5).
  --out <file>       Write the JSON report here. Otherwise stdout only.
  -h, --help         Show this help.

Eval row schema (one JSON object per line):
  { "text": "...", "entities": [{ "start": 0, "end": 4, "label": "private_person" }] }
`);
}

async function mainCli(): Promise<void> {
  const { values: args } = parseArgs({
    args: process.argv.slice(2),
    options: {
      model: { type: 'string' },
      eval: { type: 'string' },
      adapter: { type: 'string' },
      metric: { type: 'string' },
      threshold: { type: 'string' },
      out: { type: 'string' },
      help: { type: 'boolean', short: 'h', default: false },
    },
  });

  if (args.help || (!args.model && !args.eval)) {
    printHelp();
    return;
  }

  if (!args.model) {
    console.error('Error: --model is required');
    process.exit(1);
  }
  if (!args.eval) {
    console.error('Error: --eval is required');
    process.exit(1);
  }

  const modelPath = resolve(args.model);
  const evalPath = resolve(args.eval);
  const adapterPath = args.adapter ? resolve(args.adapter) : undefined;

  if (!existsSync(modelPath)) {
    console.error(`Error: model directory not found: ${modelPath}`);
    process.exit(1);
  }
  if (!existsSync(evalPath)) {
    console.error(`Error: eval file not found: ${evalPath}`);
    process.exit(1);
  }
  if (adapterPath && !existsSync(adapterPath)) {
    console.error(`Error: adapter directory not found: ${adapterPath}`);
    process.exit(1);
  }

  const metric = (args.metric ?? 'span') as 'span' | 'token';
  if (metric !== 'span' && metric !== 'token') {
    console.error(`Error: --metric must be 'span' or 'token' (got '${args.metric}')`);
    process.exit(1);
  }
  const threshold = args.threshold ? Number(args.threshold) : undefined;
  if (threshold !== undefined && !Number.isFinite(threshold)) {
    console.error(`Error: --threshold must be a finite number (got '${args.threshold}')`);
    process.exit(1);
  }

  const report = await evaluatePrivacy(modelPath, evalPath, { adapterPath, metric, threshold });
  console.log(formatReport(report));

  if (args.out) {
    const outPath = resolve(args.out);
    writeFileSync(outPath, JSON.stringify(report, null, 2));
    console.log(`\nWrote ${outPath}`);
  }
}

// Run only when invoked directly via `oxnode`. The `import.meta.url` check
// keeps `evaluatePrivacy` cleanly importable from other scripts/tests.
const isDirectInvocation = process.argv[1] !== undefined && import.meta.url === `file://${process.argv[1]}`;
if (isDirectInvocation) {
  mainCli().catch((err: unknown) => {
    console.error(err);
    process.exit(1);
  });
}
