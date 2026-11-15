// Pure scoring engine. No I/O; driven entirely by the DayRules snapshot.
import type { Action, CommitmentAnswer, DayRules } from '../types';

export interface ScoreBreakdown {
  actions: number;
  commitments: number;
  total: number;
  max: number; // best possible total under these rules
  lost: number; // max - total
}

/** Clamp a card score into 0..max; anything non-integer or missing counts as 0. */
export function cardScore(score: number | null | undefined, max: number): number {
  if (score === null || score === undefined || !Number.isFinite(score)) return 0;
  return Math.min(max, Math.max(0, Math.trunc(score)));
}

export function commitmentPoints(def: DayRules['defs'][number], a: CommitmentAnswer | undefined): number {
  if (!a || a.answer === null) return 0;
  if (a.answer === 'yes') return def.yesPts;
  if (def.kind === 'multiplier') return def.noPts * Math.max(1, Math.trunc(a.n ?? 1));
  return def.noPts;
}

export function scoreDay(
  day: { actions: Pick<Action, 'score'>[]; commitments: Record<string, CommitmentAnswer> },
  rules: DayRules,
): ScoreBreakdown {
  const actions = rules.cardMax.reduce((s, max, i) => s + cardScore(day.actions[i]?.score, max), 0);
  const commitments = rules.defs.reduce((s, def) => s + commitmentPoints(def, day.commitments[def.id]), 0);
  const max = rules.cardMax.reduce((s, m) => s + m, 0) + rules.defs.reduce((s, d) => s + Math.max(0, d.yesPts), 0);
  const total = actions + commitments;
  return { actions, commitments, total, max, lost: max - total };
}

/** Is a score valid for a card? (integer within 0..max, or empty) */
export function isValidCardScore(v: number | null, max: number): boolean {
  return v === null || (Number.isInteger(v) && v >= 0 && v <= max);
}
