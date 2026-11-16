package app.kimiagar.core

import kotlin.math.roundToInt

object Logic {
    /** Counts towards averages once anything is scored/answered (pure plans don't). */
    fun isScored(d: Day) = d.actions.any { it.score != null } || d.commitments.values.any { it.answer != null }

    fun isPlanned(d: Day) = d.intention.isNotBlank() || d.actions.any { it.topic.isNotBlank() || it.description.isNotBlank() }

    fun total(d: Day) = Scoring.scoreDay(d).total

    fun averageBetween(days: List<Day>, from: String, to: String): Double? {
        val sel = days.filter { it.date in from..to && isScored(it) }
        if (sel.isEmpty()) return null
        return (sel.sumOf { total(it) }.toDouble() / sel.size * 10).roundToInt() / 10.0
    }

    fun weekAverage(days: List<Day>, iso: String): Double? {
        val from = Dates.weekStart(iso)
        return averageBetween(days, from, Dates.add(from, 6))
    }

    fun monthAverage(days: List<Day>, iso: String): Double? {
        val (f, t) = Dates.jalaliMonthRange(iso)
        return averageBetween(days, f, t)
    }

    fun nextExperienceDue(exps: List<Experience>, courseStart: String?, interval: Int): String? {
        val base = exps.filter { it.completed && it.completedDate != null }.mapNotNull { it.completedDate }.maxOrNull() ?: courseStart
        return base?.let { Dates.add(it, interval.toLong()) }
    }

    fun daysUntil(due: String, today: String) = Dates.diff(today, due)

    fun suggestStreak(prev: CommitmentAnswer?): Int? =
        if (prev?.answer == "yes" && prev.streak != null) prev.streak + 1 else null

    fun suggestDayNumber(days: List<Day>, date: String): Int =
        (days.filter { it.date < date }.mapNotNull { it.dayNumber }.maxOrNull() ?: 0) + 1

    fun backupDue(lastBackupAt: Long?, now: Long, hasData: Boolean): Boolean =
        hasData && (lastBackupAt == null || now - lastBackupAt > 7L * 86_400_000)
}
