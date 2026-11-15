import type { CommitmentDef, DayRules } from '../types';

/** Freeze the currently active commitment definitions + card maxima into a day snapshot. */
export function buildRules(cardMax: number[], defs: CommitmentDef[]): DayRules {
  return {
    cardMax: [...cardMax],
    defs: defs
      .filter((d) => d.active)
      .sort((a, b) => a.order - b.order)
      .map(({ id, label, yesPts, noPts, kind }) => ({ id, label, yesPts, noPts, kind })),
  };
}
