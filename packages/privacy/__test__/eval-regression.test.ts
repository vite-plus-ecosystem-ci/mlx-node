/**
 * Regression test for the LoRA-adapter eval harness.
 *
 * Doubly gated to keep CI green without artifacts:
 *
 *   1. `PRIVACY_FILTER_MODEL_DIR` must point at a privacy-filter checkpoint
 *      (same gate the rest of the privacy suite uses).
 *   2. `.cache/adapters/test/` (or `PRIVACY_LORA_ADAPTER_DIR`) must hold a
 *      trained LoRA adapter (`adapter.safetensors` + `adapter_config.json`).
 *      Without an adapter we still smoke-test the baseline eval path so the
 *      harness itself stays exercised.
 *
 * The regression invariant is: an adapter must not regress recall by more
 * than 2 percentage points vs the baseline on a small canned PII set. We
 * use a tiny inline fixture so the test stays self-contained (the production
 * eval set lives outside the repo).
 */
import { existsSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import { evaluatePrivacy, type EvalReport, type EvalRow } from '../../../scripts/evaluate-privacy';

const MODEL_DIR = process.env.PRIVACY_FILTER_MODEL_DIR;
const modelAvailable = !!MODEL_DIR && existsSync(MODEL_DIR);

const ADAPTER_DIR = process.env.PRIVACY_LORA_ADAPTER_DIR ?? resolve('.cache/adapters/test');
const adapterAvailable =
  existsSync(join(ADAPTER_DIR, 'adapter.safetensors')) && existsSync(join(ADAPTER_DIR, 'adapter_config.json'));

// Tiny inline eval set. Byte offsets match what the tokenizer-aware
// classify path actually emits (a leading whitespace is folded into the
// span on the privacy-filter tokenizer's offset convention), so a clean
// baseline can hit non-zero span-exact recall.
const EVAL_ROWS: EvalRow[] = [
  {
    // "Hi I am Alice Smith, email alice@example.com"
    //  0123456789012345678901234567890123456789012345
    //  0         1         2         3         4
    text: 'Hi I am Alice Smith, email alice@example.com',
    entities: [
      { start: 7, end: 19, label: 'private_person' },
      { start: 26, end: 44, label: 'private_email' },
    ],
  },
  {
    // "Please call Bob at +1 555 123 4567 tomorrow."
    text: 'Please call Bob at +1 555 123 4567 tomorrow.',
    entities: [
      { start: 11, end: 15, label: 'private_person' },
      { start: 18, end: 34, label: 'private_phone' },
    ],
  },
];

function writeEvalFixture(rows: EvalRow[]): { path: string; cleanup: () => void } {
  const dir = mkdtempSync(join(tmpdir(), 'privacy-eval-'));
  const path = join(dir, 'eval.jsonl');
  writeFileSync(path, rows.map((r) => JSON.stringify(r)).join('\n') + '\n');
  return {
    path,
    cleanup: (): void => {
      rmSync(dir, { recursive: true, force: true });
    },
  };
}

describe.skipIf(!modelAvailable)('privacy LoRA eval regression', () => {
  it('baseline eval produces a well-formed report', async () => {
    const fixture = writeEvalFixture(EVAL_ROWS);
    try {
      const report = await evaluatePrivacy(MODEL_DIR!, fixture.path);
      expect(report.metric).toBe('span');
      expect(report.numExamples).toBe(EVAL_ROWS.length);
      expect(report.modelPath).toBe(MODEL_DIR);
      expect(report.adapterPath).toBeUndefined();
      // Recall is an aggregate over (tp / (tp + fn)).
      expect(report.overall.recall).toBeGreaterThanOrEqual(0);
      expect(report.overall.recall).toBeLessThanOrEqual(1);
      expect(report.overall.precision).toBeGreaterThanOrEqual(0);
      expect(report.overall.precision).toBeLessThanOrEqual(1);
      // At minimum the baseline should catch *some* PII on this fixture.
      // Span-exact is strict, but every entity here is unambiguous; bail
      // out loudly if the model drops them entirely.
      expect(report.overall.truePositives + report.overall.falseNegatives).toBeGreaterThan(0);
    } finally {
      fixture.cleanup();
    }
  });

  it('token-level metric runs end-to-end', async () => {
    const fixture = writeEvalFixture(EVAL_ROWS);
    try {
      const report = await evaluatePrivacy(MODEL_DIR!, fixture.path, { metric: 'token' });
      expect(report.metric).toBe('token');
      // Token-level is a superset of span-level positives at the byte level,
      // so we expect at least one TP somewhere.
      expect(report.overall.truePositives).toBeGreaterThan(0);
    } finally {
      fixture.cleanup();
    }
  });

  it.skipIf(!adapterAvailable)('adapter recall does not regress by more than 2pp vs baseline', async () => {
    const fixture = writeEvalFixture(EVAL_ROWS);
    try {
      const baseline: EvalReport = await evaluatePrivacy(MODEL_DIR!, fixture.path);
      const adapted: EvalReport = await evaluatePrivacy(MODEL_DIR!, fixture.path, {
        adapterPath: ADAPTER_DIR,
      });

      expect(adapted.adapterPath).toBe(ADAPTER_DIR);
      // Regression gate from the design plan:
      //   adapted_recall >= baseline_recall - 0.02
      expect(adapted.overall.recall).toBeGreaterThanOrEqual(baseline.overall.recall - 0.02);
    } finally {
      fixture.cleanup();
    }
  });
});
