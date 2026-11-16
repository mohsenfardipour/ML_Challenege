package app.kimiagar.core

import kotlinx.serialization.Serializable

// Field names match the desktop/web app exactly, so JSON backups move freely between phone and PC.

@Serializable
data class Settings(
    val id: String = "main",
    val name: String = "",
    val courseStart: String? = null,
    val courseLength: Int = 45,
    val cardMax: List<Int> = DEFAULT_CARD_MAX,
    val digits: String = "fa", // "fa" | "latin"
    val walkBlank: String = "",
    val expIntervalDays: Int = 14,
    val lastBackupAt: Long? = null,
)

@Serializable
data class CommitmentDef(
    val id: String,
    val order: Int,
    val label: String,
    val yesPts: Int,
    val noPts: Int,
    val kind: String, // simple | multiplier | streak | star
    val active: Boolean,
)

@Serializable
data class RuleDef(val id: String, val label: String, val yesPts: Int, val noPts: Int, val kind: String)

/** Scoring rules frozen onto each day when it is first created. */
@Serializable
data class DayRules(val cardMax: List<Int>, val defs: List<RuleDef>)

@Serializable
data class CommitmentAnswer(
    val answer: String? = null, // "yes" | "no" | null
    val n: Int? = null,
    val note: String? = null,
    val behavior: String? = null,
    val streak: Int? = null,
)

@Serializable
data class Action(
    val topic: String = "",
    val dimensionId: String? = null,
    val description: String = "",
    val reflection: String = "",
    val need: String = "",
    val value: String = "",
    val ability: String = "",
    val comfort: Int? = null, // 1..4
    val score: Int? = null,
    val status: String = "todo", // todo | done | partly | skipped
)

@Serializable
data class Day(
    val date: String, // Gregorian ISO yyyy-MM-dd
    val dayNumber: Int? = null,
    val intention: String = "",
    val actions: List<Action> = List(4) { Action() },
    val commitments: Map<String, CommitmentAnswer> = emptyMap(),
    val rules: DayRules,
    val experienceText: String = "",
    val mood: Int? = null, // 1..5
    val updatedAt: Long = 0,
)

@Serializable
data class StarExercise(
    val id: String,
    val date: String,
    val answers: Map<String, Map<String, String>> = emptyMap(),
    val current: Int = 0,
    val status: String = "draft", // draft | done
    val createdAt: Long = 0,
    val doneAt: Long? = null,
)

@Serializable
data class Experience(
    val id: String,
    val intent: String = "",
    val why: String = "",
    val targetDate: String? = null,
    val completed: Boolean = false,
    val completedDate: String? = null,
    val learned: String = "",
    val rating: Int? = null,
    val createdAt: Long = 0,
)

@Serializable
data class Dimension(val id: String, val name: String, val order: Int, val active: Boolean, val kind: String = "dimension")

@Serializable
data class DimRating(val id: String, val dimensionId: String, val date: String, val rating: Int, val note: String = "")

@Serializable
data class AppData(
    val settings: List<Settings> = listOf(Settings()),
    val commitmentDefs: List<CommitmentDef> = DEFAULT_COMMITMENTS,
    val days: List<Day> = emptyList(),
    val starExercises: List<StarExercise> = emptyList(),
    val experiences: List<Experience> = emptyList(),
    val dimensions: List<Dimension> = DEFAULT_DIMENSIONS,
    val dimRatings: List<DimRating> = emptyList(),
) {
    val s: Settings get() = settings.firstOrNull() ?: Settings()
}

@Serializable
data class BackupFile(val app: String = "kimiagar", val schema: Int = 1, val exportedAt: String, val data: AppData)
