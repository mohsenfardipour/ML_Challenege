package app.kimiagar.core

import java.time.DayOfWeek
import java.time.LocalDate

/** Port of the jalaali-js algorithm (same one the desktop app uses). Dates are stored as Gregorian ISO. */
object Jalali {
    data class JDate(val jy: Int, val jm: Int, val jd: Int)

    private val breaks = intArrayOf(-61, 9, 38, 199, 426, 686, 756, 818, 1111, 1181, 1210, 1635, 2060, 2097, 2192, 2262, 2324, 2394, 2456, 3178)

    private class Cal(val leap: Int, val gy: Int, val march: Int)

    private fun jalCal(jy: Int, withoutLeap: Boolean): Cal {
        val gy = jy + 621
        var leapJ = -14
        var jp = breaks[0]
        require(jy >= jp && jy < breaks.last()) { "Invalid Jalali year $jy" }
        var jump = 0
        for (i in 1 until breaks.size) {
            val jm = breaks[i]
            jump = jm - jp
            if (jy < jm) break
            leapJ += jump / 33 * 8 + (jump % 33) / 4
            jp = jm
        }
        var n = jy - jp
        leapJ += n / 33 * 8 + ((n % 33) + 3) / 4
        if (jump % 33 == 4 && jump - n == 4) leapJ += 1
        val leapG = gy / 4 - ((gy / 100 + 1) * 3) / 4 - 150
        val march = 20 + leapJ - leapG
        if (withoutLeap) return Cal(0, gy, march)
        if (jump - n < 6) n = n - jump + (jump + 4) / 33 * 33
        var leap = (((n + 1) % 33) - 1) % 4
        if (leap == -1) leap = 4
        return Cal(leap, gy, march)
    }

    private fun g2d(gy: Int, gm: Int, gd: Int): Int {
        var d = ((gy + (gm - 8) / 6 + 100100) * 1461) / 4 + (153 * ((gm + 9) % 12) + 2) / 5 + gd - 34840408
        d = d - ((gy + 100100 + (gm - 8) / 6) / 100 * 3) / 4 + 752
        return d
    }

    private fun d2g(jdn: Int): Triple<Int, Int, Int> {
        var j = 4 * jdn + 139361631
        j += ((4 * jdn + 183187720) / 146097 * 3) / 4 * 4 - 3908
        val i = ((j % 1461) / 4) * 5 + 308
        val gd = (i % 153) / 5 + 1
        val gm = (i / 153) % 12 + 1
        val gy = j / 1461 - 100100 + (8 - gm) / 6
        return Triple(gy, gm, gd)
    }

    private fun j2d(jy: Int, jm: Int, jd: Int): Int {
        val r = jalCal(jy, true)
        return g2d(r.gy, 3, r.march) + (jm - 1) * 31 - jm / 7 * (jm - 7) + jd - 1
    }

    private fun d2j(jdn: Int): JDate {
        val gy = d2g(jdn).first
        var jy = gy - 621
        val r = jalCal(jy, false)
        val jdn1f = g2d(gy, 3, r.march)
        var k = jdn - jdn1f
        if (k >= 0) {
            if (k <= 185) return JDate(jy, 1 + k / 31, k % 31 + 1)
            k -= 186
        } else {
            jy -= 1
            k += 179
            if (r.leap == 1) k += 1
        }
        return JDate(jy, 7 + k / 30, k % 30 + 1)
    }

    fun fromIso(iso: String): JDate {
        val d = LocalDate.parse(iso)
        return d2j(g2d(d.year, d.monthValue, d.dayOfMonth))
    }

    fun isLeap(jy: Int) = jalCal(jy, false).leap == 0

    fun monthLength(jy: Int, jm: Int) = when {
        jm <= 6 -> 31
        jm <= 11 -> 30
        isLeap(jy) -> 30
        else -> 29
    }

    fun isValid(jy: Int, jm: Int, jd: Int) = jy in -61..3177 && jm in 1..12 && jd >= 1 && jd <= monthLength(jy, jm)

    fun toIso(jy: Int, jm: Int, jd: Int): String? {
        if (!isValid(jy, jm, jd)) return null
        val (gy, gm, gd) = d2g(j2d(jy, jm, jd))
        return LocalDate.of(gy, gm, gd).toString()
    }

    val MONTHS = listOf("فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور", "مهر", "آبان", "آذر", "دی", "بهمن", "اسفند")

    fun weekdayName(iso: String): String = when (LocalDate.parse(iso).dayOfWeek) {
        DayOfWeek.SATURDAY -> "شنبه"; DayOfWeek.SUNDAY -> "یکشنبه"; DayOfWeek.MONDAY -> "دوشنبه"
        DayOfWeek.TUESDAY -> "سه‌شنبه"; DayOfWeek.WEDNESDAY -> "چهارشنبه"; DayOfWeek.THURSDAY -> "پنجشنبه"
        DayOfWeek.FRIDAY -> "جمعه"; null -> ""
    }
}

object Dates {
    fun today(): String = LocalDate.now().toString()
    fun add(iso: String, n: Long): String = LocalDate.parse(iso).plusDays(n).toString()
    fun diff(a: String, b: String): Long = LocalDate.parse(b).toEpochDay() - LocalDate.parse(a).toEpochDay()

    /** Saturday that starts the week containing iso (week = Saturday..Friday). */
    fun weekStart(iso: String): String {
        val d = LocalDate.parse(iso)
        val back = (d.dayOfWeek.value + 1) % 7 // Mon=1..Sun=7 -> Sat=6 -> 0
        return d.minusDays(back.toLong()).toString()
    }

    /** Index 0..6 of the weekday in a Saturday-first week. */
    fun satIndex(iso: String) = (LocalDate.parse(iso).dayOfWeek.value + 1) % 7

    fun jalaliMonthRange(iso: String): Pair<String, String> {
        val j = Jalali.fromIso(iso)
        return Jalali.toIso(j.jy, j.jm, 1)!! to Jalali.toIso(j.jy, j.jm, Jalali.monthLength(j.jy, j.jm))!!
    }
}
