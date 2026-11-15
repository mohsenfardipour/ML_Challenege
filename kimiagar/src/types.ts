import type { ISODate } from './lib/dates';

export type CommitmentKind = 'simple' | 'multiplier' | 'streak' | 'star';

export interface CommitmentDef {
  id: string;
  order: number;
  label: string;
  yesPts: number;
  noPts: number; // for 'multiplier' this is the per-occurrence penalty
  kind: CommitmentKind;
  active: boolean;
}

/** Scoring rules frozen onto each day when it is first created. */
export interface DayRules {
  cardMax: number[];
  defs: Omit<CommitmentDef, 'order' | 'active'>[];
}

export type Answer = 'yes' | 'no' | null;

export interface CommitmentAnswer {
  answer: Answer;
  n?: number | null; // multiplier: number of broken commitments
  note?: string; // multiplier: what I broke
  behavior?: string; // streak: behavior text
  streak?: number | null; // streak: consecutive days
}

export type ActionStatus = 'todo' | 'done' | 'partly' | 'skipped';

export interface Action {
  topic: string;
  dimensionId: string | null;
  description: string;
  reflection: string; // موانع و کشف
  need: string;
  value: string;
  ability: string;
  comfort: 1 | 2 | 3 | 4 | null; // کم .. خیلی زیاد
  score: number | null;
  status: ActionStatus;
}

export interface Day {
  date: ISODate;
  dayNumber: number | null;
  intention: string;
  actions: Action[]; // always 4
  commitments: Record<string, CommitmentAnswer>;
  rules: DayRules;
  experienceText: string;
  mood: 1 | 2 | 3 | 4 | 5 | null;
  updatedAt: number;
}

export interface Settings {
  id: 'main';
  name: string;
  courseStart: ISODate | null;
  courseLength: number;
  cardMax: number[];
  digits: 'fa' | 'latin';
  walkBlank: string;
  expIntervalDays: number;
  lastBackupAt: number | null;
}

export interface StarExercise {
  id: string;
  date: ISODate;
  answers: Record<string, Record<string, string>>; // stepId -> fieldId -> text
  current: number; // index of current step
  status: 'draft' | 'done';
  createdAt: number;
  doneAt: number | null;
}

export interface Experience {
  id: string;
  intent: string;
  why: string;
  targetDate: ISODate | null;
  completed: boolean;
  completedDate: ISODate | null;
  learned: string;
  rating: number | null; // 1..5
  createdAt: number;
}

/** kind 'dimension' = one of the 12 life dimensions; 'category' = separate activity category (e.g. کودک). */
export interface Dimension {
  id: string;
  name: string;
  order: number;
  active: boolean;
  kind: 'dimension' | 'category';
}

export interface DimRating {
  id: string;
  dimensionId: string;
  date: ISODate;
  rating: number; // 1..10
  note: string;
}
