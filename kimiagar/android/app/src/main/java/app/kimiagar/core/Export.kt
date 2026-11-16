package app.kimiagar.core

import kotlinx.serialization.json.Json
import java.time.Instant

val KJson = Json { ignoreUnknownKeys = true; encodeDefaults = true; prettyPrint = false }

object Backup {
    const val SCHEMA = 1

    fun export(data: AppData): String =
        Json { encodeDefaults = true; prettyPrint = true }.encodeToString(BackupFile.serializer(), BackupFile(exportedAt = Instant.now().toString(), data = data))

    /** Parse + validate; throws IllegalArgumentException with a Persian message. */
    fun parse(text: String): BackupFile {
        val b = try { KJson.decodeFromString(BackupFile.serializer(), text) } catch (e: Exception) {
            throw IllegalArgumentException("این فایل پشتیبان کیمیاگر نیست یا خراب است.")
        }
        require(b.app == "kimiagar") { "این فایل پشتیبان کیمیاگر نیست." }
        require(b.schema <= SCHEMA) { "این پشتیبان با نسخهٔ جدیدتری از برنامه ساخته شده است." }
        require(b.data.days.all { it.actions.isNotEmpty() && it.date.length == 10 }) { "یکی از روزها در فایل پشتیبان خراب است." }
        return b
    }
}

object Export {
    fun labelOf(label: String, s: Settings) = if (s.walkBlank.isNotBlank()) label.replace(WALK_BLANK, s.walkBlank) else label

    private fun cell(v: Any?): String {
        val s = v?.toString() ?: ""
        return if (s.any { it == '"' || it == ',' || it == '\n' || it == '\r' }) "\"" + s.replace("\"", "\"\"") + "\"" else s
    }
    private fun csv(rows: List<List<Any?>>) = "﻿" + rows.joinToString("\r\n") { r -> r.joinToString(",") { cell(it) } }
    private fun jal(iso: String) = Jalali.fromIso(iso).let { "${it.jy}/${it.jm}/${it.jd}" }

    fun daysCsv(days: List<Day>) = csv(
        listOf(listOf("date", "jalali", "day_number", "intention", "actions_score", "commitments_score", "total", "mood", "experience")) +
            days.map { d -> val s = Scoring.scoreDay(d); listOf(d.date, jal(d.date), d.dayNumber, d.intention, s.actions, s.commitments, s.total, d.mood, d.experienceText) }
    )

    fun actionsCsv(days: List<Day>, dims: List<Dimension>) = csv(
        listOf(listOf("date", "jalali", "card", "topic", "dimension", "description", "reflection", "need", "value", "ability", "comfort_zone", "score", "max", "status")) +
            days.flatMap { d ->
                d.actions.mapIndexed { i, a ->
                    listOf(d.date, jal(d.date), i + 1, a.topic, dims.find { it.id == a.dimensionId }?.name ?: "", a.description, a.reflection,
                        a.need, a.value, a.ability, a.comfort?.let { COMFORT_LABELS[it - 1] } ?: "", a.score, d.rules.cardMax.getOrNull(i), a.status)
                }
            }
    )

    fun commitmentsCsv(days: List<Day>) = csv(
        listOf(listOf("date", "jalali", "row", "label", "answer", "points", "count", "note", "behavior", "streak")) +
            days.flatMap { d ->
                d.rules.defs.mapIndexed { i, def ->
                    val a = d.commitments[def.id]
                    listOf(d.date, jal(d.date), i + 1, def.label, a?.answer ?: "", Scoring.commitmentPoints(def, a), a?.n ?: "", a?.note ?: "", a?.behavior ?: "", a?.streak ?: "")
                }
            }
    )

    fun dayAsText(d: Day, s: Settings, dims: List<Dimension>): String = buildList {
        val sc = Scoring.scoreDay(d)
        add("فرم انتخاب، ایجاد و بررسی کیمیاگر" + if (s.name.isNotBlank()) ": ${s.name}" else "")
        add("روز ${Fmt.num(d.dayNumber ?: "…")} از دوره ${Fmt.num(s.courseLength)} روزه‌ام — تاریخ: ${Fmt.jalali(d.date)}")
        if (d.intention.isNotBlank()) add("قصد امروز جاری کردن: ${d.intention}")
        d.actions.forEachIndexed { i, a ->
            add("")
            val dim = dims.find { it.id == a.dimensionId }
            add("${Fmt.num(i + 1)}) موضوع اقدام: ${a.topic.ifBlank { "—" }}" + (dim?.let { " (${it.name})" } ?: ""))
            if (a.description.isNotBlank()) add("شرح اقدام: ${a.description}")
            if (a.reflection.isNotBlank()) add("موانع و کشف: ${a.reflection}")
            listOfNotNull(a.need.takeIf { it.isNotBlank() }?.let { "نیاز: $it" }, a.value.takeIf { it.isNotBlank() }?.let { "ارزش: $it" },
                a.ability.takeIf { it.isNotBlank() }?.let { "توانایی: $it" }).takeIf { it.isNotEmpty() }?.let { add(it.joinToString(" | ")) }
            val max = d.rules.cardMax.getOrElse(i) { 0 }
            add(listOfNotNull(a.comfort?.let { "خروج از ناحیه امن: ${COMFORT_LABELS[it - 1]}" },
                a.status.takeIf { it != "todo" }?.let { STATUS_LABELS[it] },
                "امتیاز: ${Fmt.num(Scoring.cardScore(a.score, max))} از ${Fmt.num(max)}").joinToString(" | "))
        }
        add(""); add("تعهدات:")
        d.rules.defs.forEach { def ->
            val a = d.commitments[def.id]
            val mark = when (a?.answer) { "yes" -> "✅"; "no" -> "❌"; else -> "⬜" }
            var line = "$mark ${labelOf(def.label, s)}"
            if (a?.answer != null) line += " (${Fmt.signed(Scoring.commitmentPoints(def, a))})"
            if (a?.answer == "no" && def.kind == "multiplier") line += " — ${Fmt.num(a.n ?: 1)} مورد" + (a.note?.takeIf { it.isNotBlank() }?.let { ": $it" } ?: "")
            if (def.kind == "streak" && (!a?.behavior.isNullOrBlank() || a?.streak != null))
                line += " — ${a?.behavior ?: ""}" + (a?.streak?.let { " برای ${Fmt.num(it)} روز متوالی" } ?: "")
            add(line)
        }
        if (d.experienceText.isNotBlank() || d.mood != null) {
            add("")
            if (d.experienceText.isNotBlank()) add("تجربه امروز: ${d.experienceText}")
            d.mood?.let { add("حال امروز: ${MOOD_EMOJI[it - 1]} ${MOOD_TITLES[it - 1]}") }
        }
        add(""); add("امتیاز امروز: ${Fmt.num(sc.total)} (اقدام‌ها ${Fmt.num(sc.actions)} + تعهدات ${Fmt.num(sc.commitments)})")
    }.joinToString("\n")
}
