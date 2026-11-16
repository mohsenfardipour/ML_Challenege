package app.kimiagar.core

import org.junit.Assert.*
import org.junit.Test

class ScoringTest {
    private val rules = buildRules(DEFAULT_CARD_MAX, DEFAULT_COMMITMENTS)
    private val rows = rules.defs.map { it.id }

    private fun day(scores: List<Int?>, answers: Map<Int, CommitmentAnswer> = emptyMap(), defaultYes: Boolean = true): Day {
        val c = rows.mapIndexed { i, id -> id to (answers[i + 1] ?: CommitmentAnswer(if (defaultYes) "yes" else null)) }.toMap()
        return Day(date = "2026-11-15", actions = scores.map { Action(score = it) }, commitments = c, rules = rules)
    }
    private fun total(d: Day) = Scoring.scoreDay(d).total

    @Test fun twelveRowsMax100() { assertEquals(12, rows.size); assertEquals(100, Scoring.scoreDay(day(listOf(10, 10, 8, 7))).max) }
    @Test fun exampleA() { val s = Scoring.scoreDay(day(listOf(10, 10, 8, 7))); assertEquals(35, s.actions); assertEquals(65, s.commitments); assertEquals(100, s.total) }
    @Test fun exampleB() {
        val s = Scoring.scoreDay(day(listOf(10, 10, 8, 4), mapOf(1 to CommitmentAnswer("no", n = 1, note = "غیبت"))))
        assertEquals(32, s.actions); assertEquals(45, s.commitments); assertEquals(77, s.total)
    }
    @Test fun day17() = assertEquals(85, total(day(listOf(10, 10, 8, 7), mapOf(2 to CommitmentAnswer("no")))))
    @Test fun day18() = assertEquals(67, total(day(listOf(10, 10, 5, 7), mapOf(2 to CommitmentAnswer("no"), 3 to CommitmentAnswer("no")))))
    @Test fun day19() = assertEquals(88, total(day(listOf(10, 10, 8, 7), mapOf(8 to CommitmentAnswer("no"), 12 to CommitmentAnswer("no")))))
    @Test fun day23() {
        val s = Scoring.scoreDay(day(listOf(10, 10, 8, 7), mapOf(1 to CommitmentAnswer("no", n = 1), 3 to CommitmentAnswer("no"))))
        assertEquals(65, s.total); assertEquals(35, s.lost)
    }
    @Test fun multiplier() {
        assertEquals(55 - 30, Scoring.scoreDay(day(listOf(0, 0, 0, 0), mapOf(1 to CommitmentAnswer("no", n = 3)))).commitments)
        assertEquals(55 - 10, Scoring.scoreDay(day(listOf(0, 0, 0, 0), mapOf(1 to CommitmentAnswer("no", n = 0)))).commitments)
    }
    @Test fun unanswered() = assertEquals(0, total(day(listOf(null, null, null, null), defaultYes = false)))
    @Test fun clamp() { assertEquals(10, Scoring.cardScore(12, 10)); assertEquals(0, Scoring.cardScore(-3, 7)); assertEquals(35, Scoring.scoreDay(day(listOf(99, 99, 99, 99))).actions) }
}

class DatesTest {
    @Test fun known() {
        assertEquals(Jalali.JDate(1405, 8, 23), Jalali.fromIso("2026-11-14"))
        assertEquals("2026-03-21", Jalali.toIso(1405, 1, 1))
        assertEquals("2026-09-27", Jalali.toIso(1405, 7, 5))
        assertNull(Jalali.toIso(1405, 12, 30))
        assertEquals("2025-03-20", Jalali.toIso(1403, 12, 30)) // 1403 is leap
    }
    @Test fun matchesJsLibrary() {
        // Vectors generated from jalaali-js (the library the desktop app uses): gy,gm,gd,jy,jm,jd
        "1990,1,2,1368,10,12;1992,9,6,1371,6,15;1994,6,23,1373,4,2;1995,5,21,1374,2,31;1995,5,30,1374,3,9;1998,2,9,1376,11,20;1999,12,4,1378,9,13;2000,11,8,1379,8,18;2000,11,25,1379,9,5;2003,8,16,1382,5,25;2005,6,17,1384,3,27;2006,5,31,1385,3,10;2006,6,25,1385,4,4;2009,3,23,1388,1,3;2011,1,31,1389,11,11;2012,1,22,1390,11,2;2012,2,24,1390,12,5;2014,11,30,1393,9,9;2016,10,17,1395,7,26;2017,10,16,1396,7,24;2017,11,26,1396,9,5;2020,9,9,1399,6,19;2022,8,5,1401,5,14;2023,8,12,1402,5,21;2023,9,30,1402,7,8;2026,7,22,1405,4,31;2028,6,24,1407,4,4;2029,7,9,1408,4,19;2029,9,4,1408,6,14;2032,7,4,1411,4,14;2034,6,15,1413,3,25;2035,7,8,1414,4,17;2035,9,11,1414,6,20;2038,7,19,1417,4,28;2040,7,7,1419,4,17;2041,8,7,1420,5,17;2041,10,19,1420,7,28;2044,9,3,1423,6,13;2046,8,31,1425,6,9;2047,10,9,1426,7,17;2047,12,29,1426,10,8;2050,11,21,1429,8,30;2052,11,25,1431,9,5;2054,1,11,1432,10,22;2054,4,10,1433,1,21;2057,3,11,1435,12,21;2059,3,24,1438,1,4;2060,5,17,1439,2,28;2060,8,22,1439,6,1;2063,8,1,1442,5,10;2065,8,21,1444,5,31;2066,10,23,1445,8,2;2067,2,5,1445,11,17;2070,1,22,1448,11,3;2072,2,20,1450,12,1;2073,5,1,1452,2,12;2073,8,22,1452,6,1;2076,8,16,1455,5,26;2078,9,22,1457,7,1;2079,12,10,1458,9,19".split(";").forEach { v ->
            val (gy, gm, gd, jy, jm, jd) = v.split(",").map { it.toInt() }
            val iso = "%04d-%02d-%02d".format(gy, gm, gd)
            assertEquals(v, Jalali.JDate(jy, jm, jd), Jalali.fromIso(iso))
            assertEquals(v, iso, Jalali.toIso(jy, jm, jd))
        }
    }
    @Test fun weekStartsSaturday() {
        assertEquals("2026-11-14", Dates.weekStart("2026-11-14"))
        assertEquals("2026-11-14", Dates.weekStart("2026-11-20"))
        assertEquals("2026-11-21", Dates.weekStart("2026-11-21"))
    }
    private operator fun <T> List<T>.component6() = this[5]
}

class LogicTest {
    private val rules = buildRules(DEFAULT_CARD_MAX, DEFAULT_COMMITMENTS)
    private fun d(date: String, scores: List<Int?>, n: Int? = null) = Day(date, dayNumber = n, actions = scores.map { Action(score = it) }, rules = rules)
    private fun exp(c: String?) = Experience(id = c ?: "x", completed = c != null, completedDate = c)

    @Test fun averages() {
        val days = listOf(d("2026-11-14", listOf(10, 10, 8, 7)), d("2026-11-15", listOf(10, 10, 0, 0)), d("2026-11-16", listOf(null, null, null, null)))
        assertEquals(27.5, Logic.averageBetween(days, "2026-11-01", "2026-11-30")!!, 0.0)
        assertEquals(27.5, Logic.weekAverage(days, "2026-11-20")!!, 0.0)
        assertNull(Logic.weekAverage(days, "2026-11-13"))
    }
    @Test fun experienceCycle() {
        assertEquals("2026-11-15", Logic.nextExperienceDue(emptyList(), "2026-11-01", 14))
        assertEquals("2026-11-24", Logic.nextExperienceDue(listOf(exp("2026-11-10"), exp("2026-11-03"), exp(null)), "2026-11-01", 14))
        assertEquals("2026-11-20", Logic.nextExperienceDue(listOf(exp("2026-11-10")), null, 10))
        assertNull(Logic.nextExperienceDue(emptyList(), null, 14))
        assertEquals(5, Logic.daysUntil("2026-11-20", "2026-11-15"))
        assertEquals(-5, Logic.daysUntil("2026-11-10", "2026-11-15"))
    }
    @Test fun suggestions() {
        assertEquals(7, Logic.suggestStreak(CommitmentAnswer("yes", streak = 6)))
        assertNull(Logic.suggestStreak(CommitmentAnswer("no", streak = 6)))
        assertNull(Logic.suggestStreak(null))
        val days = listOf(d("2026-10-01", emptyList(), 20), d("2026-10-11", emptyList(), 23))
        assertEquals(24, Logic.suggestDayNumber(days, "2026-10-22"))
        assertEquals(21, Logic.suggestDayNumber(days, "2026-10-05"))
        assertEquals(1, Logic.suggestDayNumber(emptyList(), "2026-10-05"))
    }
    @Test fun backupReminder() {
        val now = System.currentTimeMillis()
        assertTrue(Logic.backupDue(null, now, true)); assertFalse(Logic.backupDue(null, now, false))
        assertFalse(Logic.backupDue(now - 3 * 86_400_000L, now, true)); assertTrue(Logic.backupDue(now - 8 * 86_400_000L, now, true))
    }
}

class BackupTest {
    @Test fun roundTripAndWebCompat() {
        val rules = buildRules(DEFAULT_CARD_MAX, DEFAULT_COMMITMENTS)
        val day = Day("2026-09-27", 16, "عشق و محبت", List(4) { Action(topic = "خانواده", score = 10) }, mapOf("c01" to CommitmentAnswer("no", n = 1, note = "غیبت")), rules)
        val data = AppData(days = listOf(day))
        val back = Backup.parse(Backup.export(data))
        assertEquals(data, back.data)
        assertTrue(Export.dayAsText(day, Settings(), emptyList()).contains("غیبت"))
        assertTrue(Export.daysCsv(listOf(day)).startsWith("﻿"))
    }
    @Test fun rejectsForeign() {
        assertThrows(IllegalArgumentException::class.java) { Backup.parse("{\"hello\":1}") }
        assertThrows(IllegalArgumentException::class.java) { Backup.parse("not json") }
    }
}

class WebBackupCompatTest {
    @Test fun restoresBackupMadeByDesktopApp() {
        val text = javaClass.getResource("/web-backup.json")!!.readText()
        val b = Backup.parse(text)
        assertEquals(2, b.data.days.size)
        assertEquals(1, b.data.starExercises.size)
        val scored = b.data.days.first { Logic.isScored(it) }
        assertEquals(77, Scoring.scoreDay(scored).total)
        assertEquals("غیبت", scored.commitments["c01"]?.note)
    }
}
