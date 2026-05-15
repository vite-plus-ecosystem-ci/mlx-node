/**
 * Tests for the HuggingFace-compatible `aggregationStrategy` option on
 * {@link PrivacyFilter.classify}. Mirrors HF's
 * `TokenClassificationPipeline.aggregation_strategy` behavior.
 *
 * Gated on `PRIVACY_FILTER_MODEL_DIR` so CI without weights stays green —
 * same pattern as `parity.test.ts` / `classify.test.ts`.
 */
import { existsSync } from 'node:fs';

import { PrivacyFilter } from '@mlx-node/privacy';
import type { AggregationStrategy } from '@mlx-node/privacy';
import { describe, expect, it } from 'vite-plus/test';

const MODEL_DIR = process.env.PRIVACY_FILTER_MODEL_DIR;
const modelAvailable = !!MODEL_DIR && existsSync(MODEL_DIR);

const PII_INPUT = 'Hi I am Alice Smith, email alice@example.com';

describe.skipIf(!modelAvailable)('PrivacyFilter.classify aggregationStrategy', () => {
  it('"none" returns ≥2 per-token entities labeled with full BIOES tags', async () => {
    const pf = await PrivacyFilter.load(MODEL_DIR!);
    const { entities } = await pf.classify(PII_INPUT, { aggregationStrategy: 'none' });
    expect(entities.length).toBeGreaterThanOrEqual(2);
    // Every label should be a BIOES tag — must contain '-' (e.g. 'B-private_email').
    for (const e of entities) {
      expect(e.label).toMatch(/^[BIES]-[a-z_]+$/);
    }
    // At least one entity must be a private_email token (B- or S-).
    // The wrapper narrows `label` to `PrivacyLabel` (no BIOES prefix);
    // on the `aggregationStrategy: 'none'` path it actually carries the
    // full BIOES tag, so widen back to `string` for this comparison.
    const hasEmail = entities.some((e) => {
      const label: string = e.label;
      return label === 'B-private_email' || label === 'S-private_email';
    });
    expect(hasEmail).toBe(true);
  });

  it('"simple" returns entities with stripped class labels (no BIOES prefix)', async () => {
    const pf = await PrivacyFilter.load(MODEL_DIR!);
    const { entities } = await pf.classify(PII_INPUT, { aggregationStrategy: 'simple' });
    expect(entities.length).toBeGreaterThan(0);
    for (const e of entities) {
      // Stripped labels never contain a BIOES dash prefix.
      expect(e.label).not.toMatch(/^[BIES]-/);
    }
  });

  it('"first" / "average" / "max" all produce stripped class labels', async () => {
    const pf = await PrivacyFilter.load(MODEL_DIR!);
    for (const strategy of ['first', 'average', 'max'] as const) {
      const { entities } = await pf.classify(PII_INPUT, { aggregationStrategy: strategy });
      expect(entities.length).toBeGreaterThan(0);
      for (const e of entities) {
        expect(e.label).not.toMatch(/^[BIES]-/);
      }
    }
  });

  it('"simple" and default Viterbi cover the same byte ranges with the same labels', async () => {
    // Both paths emit overlapping coverage for the same PII spans, but
    // HF "simple" fragments multi-token spans (the `get_tag` quirk turns
    // `B-X I-X E-X` into two groups), while the BIOES Viterbi path
    // emits a single span. We assert weaker but meaningful invariants:
    //   1. The set of class labels matches exactly.
    //   2. Every byte range covered by Viterbi is covered by `simple`
    //      (and vice versa) — i.e. they agree on where PII is.
    const pf = await PrivacyFilter.load(MODEL_DIR!);
    const viterbi = await pf.classify(PII_INPUT);
    const simple = await pf.classify(PII_INPUT, { aggregationStrategy: 'simple' });

    const labels = (es: typeof viterbi.entities) => [...new Set(es.map((e) => e.label))].sort();
    expect(labels(simple.entities)).toEqual(labels(viterbi.entities));

    const covered = (es: typeof viterbi.entities, byte: number) => es.some((e) => e.start <= byte && byte < e.end);

    // Sample every byte covered by either side; both sides must agree.
    const allBytes = new Set<number>();
    for (const e of [...viterbi.entities, ...simple.entities]) {
      for (let b = e.start; b < e.end; b++) allBytes.add(b);
    }
    for (const b of allBytes) {
      expect(covered(simple.entities, b), `byte ${b}`).toBe(covered(viterbi.entities, b));
    }
  });

  it('returnTokens=true with aggregationStrategy returns raw-argmax BIOES tags', async () => {
    const pf = await PrivacyFilter.load(MODEL_DIR!);
    const { tokens } = await pf.classify(PII_INPUT, {
      aggregationStrategy: 'simple',
      returnTokens: true,
    });
    expect(tokens).toBeDefined();
    expect(tokens!.length).toBeGreaterThan(0);
    expect(tokens!.some((t) => t.tag !== 'O')).toBe(true);
  });

  it('rejects an invalid aggregationStrategy with a clear error', async () => {
    const pf = await PrivacyFilter.load(MODEL_DIR!);
    await expect(
      pf.classify(PII_INPUT, {
        aggregationStrategy: 'bogus' as AggregationStrategy,
      }),
    ).rejects.toThrow(/aggregationStrategy/);
  });
});
