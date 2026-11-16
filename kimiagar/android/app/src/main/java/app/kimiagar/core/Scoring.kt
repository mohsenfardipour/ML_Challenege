package app.kimiagar.core

/** Pure scoring engine, identical rules to the desktop app. */
data class ScoreBreakdown(val actions: Int, val commitments: Int, val total: Int, val max: Int, val lost: Int)

object Scoring {
    fun cardScore(score: Int?, max: Int): Int = if (score == null) 0 else score.coerceIn(0, max)

    fun commitmentPoints(def: RuleDef, a: CommitmentAnswer?): Int = when (a?.answer) {
        "yes" -> def.yesPts
        "no" -> if (def.kind == "multiplier") def.noPts * maxOf(1, a.n ?: 1) else def.noPts
        else -> 0
    }

    fun scoreDay(actions: List<Action>, commitments: Map<String, CommitmentAnswer>, rules: DayRules): ScoreBreakdown {
        val act = rules.cardMax.withIndex().sumOf { (i, max) -> cardScore(actions.getOrNull(i)?.score, max) }
        val com = rules.defs.sumOf { commitmentPoints(it, commitments[it.id]) }
        val max = rules.cardMax.sum() + rules.defs.sumOf { maxOf(0, it.yesPts) }
        val total = act + com
        return ScoreBreakdown(act, com, total, max, max - total)
    }

    fun scoreDay(d: Day) = scoreDay(d.actions, d.commitments, d.rules)
}
