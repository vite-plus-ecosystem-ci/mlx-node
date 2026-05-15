/**
 * Public types for the `@mlx-node/privacy` package.
 *
 * Mirrors the native NAPI surface ({@link import('@mlx-node/core').PrivacyEntity}
 * etc.) but narrows the `label` field to the literal union of the 8 PII
 * classes produced by the privacy-filter checkpoint, so consumers get
 * autocomplete on `e.label === 'private_email'`.
 */

/**
 * The 8 PII categories the privacy-filter checkpoint can emit (without the
 * BIOES prefix). At runtime the native binding returns a plain `string`, but
 * any non-background span produced by the model is guaranteed to be one of
 * these values, so the wrapper narrows it for callers.
 */
export type PrivacyLabel =
  | 'account_number'
  | 'private_address'
  | 'private_date'
  | 'private_email'
  | 'private_person'
  | 'private_phone'
  | 'private_url'
  | 'secret';

/**
 * A detected PII span. `start`/`end` are byte offsets into the original
 * input string (Hugging Face `tokenizers` convention). `score` is the mean —
 * across the span's tokens — of the softmax probability of the Viterbi-emitted
 * tag at each token.
 */
export interface Entity {
  start: number;
  end: number;
  label: PrivacyLabel;
  score: number;
  text: string;
}

/**
 * Per-call Viterbi calibration overrides.
 *
 * Field names match the native NAPI binding's camelCase casing
 * ({@link import('@mlx-node/core').PrivacyCalibration}) so the wrapper can
 * pass user options straight through without re-keying.
 *
 * Any omitted field falls back to the model's default calibration
 * (loaded from `viterbi_calibration.json` at load time).
 */
export interface ViterbiCalibration {
  transitionBiasBackgroundStay: number;
  transitionBiasBackgroundToStart: number;
  transitionBiasEndToBackground: number;
  transitionBiasEndToStart: number;
  transitionBiasInsideToContinue: number;
  transitionBiasInsideToEnd: number;
}

/**
 * Hugging Face `TokenClassificationPipeline.aggregation_strategy` values.
 *
 * - `'none'`: emit one entity per non-`O` token; `label` retains the full
 *   BIOES tag (e.g. `'B-private_email'`).
 * - `'simple'`: per-token argmax → HF `group_entities`. Note the HF quirk:
 *   `E-X` and `S-X` are treated as their own opaque tag rather than as
 *   continuations of `X`, so multi-token spans tagged `B-X I-X E-X`
 *   produce two groups (`[B-X, I-X]` and `[E-X]`) rather than one. This
 *   is byte-for-byte HF behavior, not a bug.
 * - `'first'`: aggregate subwords to words by taking the first subword's
 *   scores, then `group_entities`.
 * - `'average'`: aggregate subwords by element-wise mean of score vectors.
 * - `'max'`: pick the subword with the largest peak score.
 */
export type AggregationStrategy = 'none' | 'simple' | 'first' | 'average' | 'max';

/**
 * Options for {@link PrivacyFilter.classify}.
 *
 * - `threshold` (default `0.5`): minimum mean per-token probability for an
 *   extracted span/group to be returned.
 * - `calibration`: per-call overrides on top of the checkpoint default.
 *   **Ignored when `aggregationStrategy` is set** (HF aggregation paths
 *   skip the Viterbi decoder entirely).
 * - `returnTokens` (default `false`): when `true`, the result includes a
 *   `tokens` array with one entry per input token. With
 *   `aggregationStrategy`, the per-token `tag` is the raw argmax BIOES
 *   tag and `score` is the argmax probability.
 * - `aggregationStrategy`: when set, skip the constrained BIOES Viterbi
 *   decoder and replicate Hugging Face's
 *   `TokenClassificationPipeline.aggregation_strategy` behavior on raw
 *   per-token softmax. See {@link AggregationStrategy}.
 */
export interface ClassifyOptions {
  threshold?: number;
  calibration?: Partial<ViterbiCalibration>;
  returnTokens?: boolean;
  aggregationStrategy?: AggregationStrategy;
}

/**
 * A single token with its Viterbi-decoded tag. Emitted by
 * {@link PrivacyFilter.classify} when `returnTokens: true`.
 *
 * `tag` is the full BIOES tag (`'O'` or `'B-...'`/`'I-...'`/`'E-...'`/
 * `'S-...'`) chosen by the Viterbi decoder. `score` is the softmax
 * probability of that emitted tag at this token, so `tag` and `score`
 * always share decoders (at boundary tokens the Viterbi pick can differ
 * from the local argmax).
 */
export interface Token {
  text: string;
  tag: string;
  score: number;
  start: number;
  end: number;
}

/** Result of {@link PrivacyFilter.classify}. */
export interface ClassifyResult {
  entities: Entity[];
  tokens?: Token[];
}

/**
 * How to replace a detected entity in {@link PrivacyFilter.redact}.
 *
 * - The sentinel string `'label'` produces `[<label>]` (this is the
 *   default when no `replacement` is provided).
 * - Any other string is inserted verbatim.
 * - A function receives the entity and returns the replacement string.
 */
export type Replacement = string | ((entity: Entity) => string);

/**
 * Options for {@link PrivacyFilter.redact}. Extends {@link ClassifyOptions}
 * so threshold / calibration / returnTokens can be passed through to the
 * underlying classify call.
 */
export interface RedactOptions extends ClassifyOptions {
  /** How to replace each entity. Defaults to `'label'`. */
  replacement?: Replacement;
  /** If set, only entities whose label is in this list are redacted. */
  labels?: PrivacyLabel[];
}

/** Result of {@link PrivacyFilter.redact}. */
export interface RedactResult {
  /** The input text with each entity span replaced. */
  redacted: string;
  /** The entities that were actually redacted (after `labels` filter). */
  entities: Entity[];
}
