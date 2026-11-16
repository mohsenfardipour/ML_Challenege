package app.kimiagar.core

val DEFAULT_CARD_MAX = listOf(10, 10, 8, 7)
const val WALK_BLANK = "........"

// Exactly as printed on the workbook. Row 5 (affirmation) replaced by Star Rewriting per the owner; original kept inactive.
val DEFAULT_COMMITMENTS = listOf(
    CommitmentDef("c01", 1, "متعهد بوده‌ام؟ (هر عدم تعهد ۱۰- امتیاز)", 10, -10, "multiplier", true),
    CommitmentDef("c02", 2, "به موقع بیدار شده‌ام؟", 5, -10, "simple", true),
    CommitmentDef("c03", 3, "۱۵ دقیقه پیاده‌روی در روز با موزیک مثبت و $WALK_BLANK داشته‌ام؟", 5, -10, "simple", true),
    CommitmentDef("c04", 4, "یک عمل محبت‌آمیز یا دو مورد تحسین دیگران داشته‌ام؟", 5, 0, "simple", true),
    CommitmentDef("c13", 5, "ستاره بازنویسی را تکمیل کرده‌ام؟", 5, 0, "star", true),
    CommitmentDef("c06", 6, "ملاقات صبح‌گاهی داشته‌ام WOOP + تلقین", 5, -5, "simple", true),
    CommitmentDef("c07", 7, "حداقل برای ۱۰ مورد شکرگزاری روزانه کرده‌ام؟", 5, 0, "simple", true),
    CommitmentDef("c08", 8, "حداقل ۳ صفحه کتابم را مطالعه کرده‌ام؟", 5, -2, "simple", true),
    CommitmentDef("c09", 9, "حداقل ۲ دقیقه خودم را تحسین کرده‌ام؟", 5, 0, "simple", true),
    CommitmentDef("c10", 10, "رفتار غیرسازنده‌ای را حذف کرده‌ام؟", 5, 0, "streak", true),
    CommitmentDef("c11", 11, "رفتار سازنده‌ام را انجام داده‌ام؟", 5, 0, "streak", true),
    CommitmentDef("c12", 12, "با ارسال این فرم کامل شده‌ام؟", 5, 0, "simple", true),
    CommitmentDef("c05", 13, "قصد یا عبارت سازنده تأکیدی‌ام را تکرار کرده‌ام؟", 5, 0, "simple", false),
)

val DEFAULT_DIMENSIONS: List<Dimension> = listOf(
    "خانواده", "دوستان", "رابطه عاطفی", "سلامت جسم", "سلامت روان", "شغل و حرفه",
    "آموزش و یادگیری", "سلامت مالی", "معنویت", "روابط اجتماعی", "تفریح و اوقات فراغت", "رشد فردی و خودآگاهی",
).mapIndexed { i, n -> Dimension("d%02d".format(i + 1), n, i + 1, true, "dimension") } +
    Dimension("k01", "کودک", 13, true, "category")

val COMFORT_LABELS = listOf("کم", "متوسط", "زیاد", "خیلی زیاد")
val STATUS_LABELS = linkedMapOf("todo" to "انجام نشده", "done" to "انجام شد", "partly" to "نیمه‌کاره", "skipped" to "رها شد")
val MOOD_TITLES = listOf("خیلی خوب", "خوب", "معمولی", "ناراحت", "آشفته")
val MOOD_EMOJI = listOf("😀", "🙂", "😐", "🙁", "😧")

fun buildRules(cardMax: List<Int>, defs: List<CommitmentDef>) = DayRules(
    cardMax = cardMax.toList(),
    defs = defs.filter { it.active }.sortedBy { it.order }.map { RuleDef(it.id, it.label, it.yesPts, it.noPts, it.kind) },
)
