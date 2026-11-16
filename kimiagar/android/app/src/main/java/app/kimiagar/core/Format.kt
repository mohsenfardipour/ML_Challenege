package app.kimiagar.core

object Fmt {
    @Volatile var persianDigits = true
    private const val FA = "۰۱۲۳۴۵۶۷۸۹"

    fun num(v: Any?): String {
        if (v == null) return ""
        val s = v.toString().removeSuffix(".0")
        val out = if (persianDigits) s.map { if (it in '0'..'9') FA[it - '0'] else it }.joinToString("") else s
        // LRM keeps a leading minus sign on the left of the digits inside right-to-left text (like the paper form).
        return if (out.startsWith("-")) "$LRM$out" else out
    }

    private const val LRM = "\u200E"

    fun signed(v: Int) = if (v > 0) "$LRM+${num(v)}" else num(v)

    /** Persian/Arabic/Latin digits -> Int, or null. */
    fun parseInt(s: String): Int? {
        val latin = s.replace(LRM, "").trim().map {
            when (it) {
                in '۰'..'۹' -> '0' + (it - '۰')
                in '٠'..'٩' -> '0' + (it - '٠')
                '−', '–' -> '-'
                else -> it
            }
        }.joinToString("")
        return if (Regex("^-?\\d{1,6}$").matches(latin)) latin.toInt() else null
    }

    fun jalali(iso: String): String {
        val j = Jalali.fromIso(iso)
        return num("%d/%02d/%02d".format(j.jy, j.jm, j.jd))
    }

    fun jalaliLong(iso: String): String {
        val j = Jalali.fromIso(iso)
        return "${Jalali.weekdayName(iso)} ${num(j.jd)} ${Jalali.MONTHS[j.jm - 1]} ${num(j.jy)}"
    }
}
