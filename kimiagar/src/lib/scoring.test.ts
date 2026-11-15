import { describe, expect, it } from 'vitest';
import { scoreDay, cardScore } from './scoring';
import { buildRules } from './rules';
import { DEFAULT_CARD_MAX, DEFAULT_COMMITMENTS } from '../defaults';
import type { CommitmentAnswer } from '../types';

const rules = buildRules(DEFAULT_CARD_MAX, DEFAULT_COMMITMENTS);
// Active rows in printed order (row 5 is now Star Rewriting with the same +5/0 points).
const ROWS = rules.defs.map((d) => d.id);

function day(scores: (number | null)[], answers: Record<number, CommitmentAnswer>, defaultYes = true) {
  const commitments: Record<string, CommitmentAnswer> = {};
  ROWS.forEach((id, i) => {
    commitments[id] = answers[i + 1] ?? { answer: defaultYes ? 'yes' : null };
  });
  return { actions: scores.map((score) => ({ score })), commitments };
}

describe('scoring engine (real workbook pages)', () => {
  it('has 12 active rows and a max of 100', () => {
    expect(ROWS).toHaveLength(12);
    expect(scoreDay(day([10, 10, 8, 7], {}), rules).max).toBe(100);
  });

  it('Example A: 10,10,8,7 + all Yes = 100', () => {
    const s = scoreDay(day([10, 10, 8, 7], {}), rules);
    expect(s).toMatchObject({ actions: 35, commitments: 65, total: 100, lost: 0 });
  });

  it('Example B (day 16): 10,10,8,4 + row 1 No n=1 = 77', () => {
    const s = scoreDay(day([10, 10, 8, 4], { 1: { answer: 'no', n: 1, note: 'غیبت' } }), rules);
    expect(s).toMatchObject({ actions: 32, commitments: 45, total: 77 });
  });

  it('Day 17: row 2 No = 85', () => {
    expect(scoreDay(day([10, 10, 8, 7], { 2: { answer: 'no' } }), rules).total).toBe(85);
  });

  it('Day 18 (owner-confirmed): 10,10,5,7 + rows 2,3 No = 67', () => {
    expect(scoreDay(day([10, 10, 5, 7], { 2: { answer: 'no' }, 3: { answer: 'no' } }), rules).total).toBe(67);
  });

  it('Day 19: row 8 No (-2), row 12 No (0) = 88', () => {
    expect(scoreDay(day([10, 10, 8, 7], { 8: { answer: 'no' }, 12: { answer: 'no' } }), rules).total).toBe(88);
  });

  it('Day 23: row 1 No n=1, row 3 No = 65', () => {
    const s = scoreDay(day([10, 10, 8, 7], { 1: { answer: 'no', n: 1 }, 3: { answer: 'no' } }), rules);
    expect(s.total).toBe(65);
    expect(s.lost).toBe(35); // matches the «۳۵-» tally in the margin
  });

  it('row 1 No multiplies by n (min 1)', () => {
    expect(scoreDay(day([0, 0, 0, 0], { 1: { answer: 'no', n: 3 } }), rules).commitments).toBe(55 - 30);
    expect(scoreDay(day([0, 0, 0, 0], { 1: { answer: 'no', n: 0 } }), rules).commitments).toBe(55 - 10);
  });

  it('unanswered rows and empty cards count 0', () => {
    expect(scoreDay(day([null, null, null, null], {}, false), rules).total).toBe(0);
  });

  it('card scores are clamped to 0..max', () => {
    expect(cardScore(12, 10)).toBe(10);
    expect(cardScore(-3, 7)).toBe(0);
    expect(scoreDay(day([99, 99, 99, 99], {}), rules).actions).toBe(35);
  });

  it('a day keeps its own rule snapshot when settings change later', () => {
    const changed = buildRules([10, 10, 8, 7], DEFAULT_COMMITMENTS.map((d) => (d.id === 'c02' ? { ...d, noPts: -20 } : d)));
    const d = day([10, 10, 8, 7], { 2: { answer: 'no' } });
    expect(scoreDay(d, rules).total).toBe(85);
    expect(scoreDay(d, changed).total).toBe(75);
  });
});
