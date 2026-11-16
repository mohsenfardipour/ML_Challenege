@file:OptIn(ExperimentalLayoutApi::class)

package app.kimiagar.ui

import android.content.Intent
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import app.kimiagar.core.*
import app.kimiagar.data.Repo

const val LEAD_IN = "برای اینکه به هدفم برسم، امروز باید چه قدمی بردارم؟"
const val REFLECTION_Q = "چگونه از قصدم برای انجام این اقدام استفاده کردم؟"

data class Suggestions(val topics: List<String>, val need: List<String>, val value: List<String>, val ability: List<String>)

private fun frequent(values: List<String>): List<String> =
    values.flatMap { it.split('،', ',', '/') }.map { it.trim() }.filter { it.isNotEmpty() }
        .groupingBy { it }.eachCount().entries.sortedByDescending { it.value }.map { it.key }.take(6)

fun suggestionsFor(days: List<Day>, exclude: String): Suggestions {
    val acts = days.filter { it.date != exclude }.flatMap { it.actions }
    return Suggestions(
        topics = acts.asReversed().map { it.topic.trim() }.filter { it.isNotEmpty() }.distinct().take(6),
        need = frequent(acts.map { it.need }), value = frequent(acts.map { it.value }), ability = frequent(acts.map { it.ability }),
    )
}

@Composable
fun DayScreen(date: String, startInPlan: Boolean) {
    val data by Repo.data.collectAsState()
    val s = data.s
    var day by remember(date) { mutableStateOf(Repo.dayOrBlank(date)) }
    fun set(d: Day) { day = d; Repo.putDay(d) }
    val prev = remember(date) { Repo.prevDay(date) }
    val sugg = remember(date) { suggestionsFor(data.days, date) }
    var plan by remember(date) { mutableStateOf(startInPlan || date > today()) }
    val clipboard = LocalClipboardManager.current
    val ctx = LocalContext.current
    var confirmRules by remember { mutableStateOf(false) }

    val rel = when (date) { today() -> "امروز · "; Dates.add(today(), 1) -> "فردا · "; Dates.add(today(), -1) -> "دیروز · "; else -> "" }
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.SpaceBetween) {
        TextButton({ Nav.stack[Nav.stack.lastIndex] = Screen.DayPage(Dates.add(date, -1)) }) { Text("→", fontSize = 20.sp) }
        Text(rel + Fmt.jalaliLong(date), fontWeight = FontWeight.Bold)
        TextButton({ Nav.stack[Nav.stack.lastIndex] = Screen.DayPage(Dates.add(date, 1)) }) { Text("←", fontSize = 20.sp) }
    }
    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        Pill("برنامه‌ریزی (شب قبل)", plan, Modifier.weight(1f)) { plan = true }
        Pill("انجام و بررسی", !plan, Modifier.weight(1f)) { plan = false }
    }

    KCard {
        Text("فرم انتخاب، ایجاد و بررسی کیمیاگر: " + s.name.ifBlank { "(نام در تنظیمات)" }, fontWeight = FontWeight.Bold)
        FlowRow(verticalArrangement = Arrangement.Center, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            Text("روز", Modifier.align(Alignment.CenterVertically))
            IntField(day.dayNumber, "dn$date", 1, 999) { set(day.copy(dayNumber = it)) }
            Text("از دوره ${Fmt.num(s.courseLength)} روزه‌ام.   تاریخ: ${Fmt.jalali(date)}", Modifier.align(Alignment.CenterVertically))
        }
        BoundField(day.intention, "قصد امروز جاری کردن", "int$date") { set(day.copy(intention = it)) }
    }

    day.actions.forEachIndexed { i, a ->
        ActionCard(i, a, day.rules.cardMax.getOrElse(i) { 0 }, plan, data.dimensions, sugg, prev?.actions?.getOrNull(i)?.topic, date) { na ->
            set(day.copy(actions = day.actions.mapIndexed { j, x -> if (j == i) na else x }))
        }
    }

    if (!plan) {
        KCard { Commitments(day, s, prev, compact = false) { id, ca -> set(day.copy(commitments = day.commitments + (id to ca))) } }
        KCard {
            BoundField(day.experienceText, "تجربه امروز", "exp$date", minLines = 3) { set(day.copy(experienceText = it)) }
            Spacer(Modifier.height(6.dp))
            MoodPicker(day.mood) { set(day.copy(mood = it)) }
        }
        ScorePanel(day)
    }

    FlowRow(Modifier.padding(vertical = 8.dp), horizontalArrangement = Arrangement.spacedBy(6.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
        if (plan) PrimaryButton("رفتن به انجام و بررسی") { plan = false }
        else PrimaryButton("برنامه‌ریزی فردا") { Nav.go(Screen.DayPage(Dates.add(date, 1), plan = true)) }
        GhostButton("کپی متن روز") { clipboard.setText(AnnotatedString(Export.dayAsText(day, s, data.dimensions))) }
        GhostButton("ارسال (تلگرام…)") {
            val i = Intent(Intent.ACTION_SEND).setType("text/plain").putExtra(Intent.EXTRA_TEXT, Export.dayAsText(day, s, data.dimensions))
            ctx.startActivity(Intent.createChooser(i, "ارسال متن روز"))
        }
        TextButton({ confirmRules = true }) { Text("اعمال قوانین فعلی", color = K.muted) }
    }
    if (confirmRules) AlertDialog(
        onDismissRequest = { confirmRules = false },
        title = { Text("اعمال قوانین فعلی") },
        text = { Text("قوانین امتیازدهی فعلی (از تنظیمات) روی این روز اعمال شود؟ امتیاز این روز ممکن است تغییر کند.") },
        confirmButton = { TextButton({ set(day.copy(rules = buildRules(s.cardMax, data.commitmentDefs))); confirmRules = false }) { Text("بله") } },
        dismissButton = { TextButton({ confirmRules = false }) { Text("خیر") } },
    )
}

@Composable
fun ActionCard(i: Int, a: Action, max: Int, plan: Boolean, dims: List<Dimension>, sugg: Suggestions, prevTopic: String?, date: String, onChange: (Action) -> Unit) {
    val k = "$date-$i"
    KCard(border = if (a.status == "done") K.good else K.ink) {
        Row(verticalAlignment = Alignment.Top) {
            Box(Modifier.padding(top = 4.dp, end = 0.dp).size(26.dp).clip(CircleShape).background(K.accent), contentAlignment = Alignment.Center) {
                Text(Fmt.num(i + 1), color = Color.White, fontSize = 13.sp)
            }
            Spacer(Modifier.width(8.dp))
            Column(Modifier.weight(1f)) {
                if (i < 2) Text(LEAD_IN, fontWeight = FontWeight.Bold, fontSize = 14.sp)
                BoundField(a.topic, "موضوع اقدام", "t$k") { onChange(a.copy(topic = it)) }
                if (a.topic.isBlank()) Chips((listOfNotNull(prevTopic?.takeIf { it.isNotBlank() }) + sugg.topics).distinct().take(5)) { onChange(a.copy(topic = it)) }
            }
        }
        BoundField(a.description, "شرح اقدام", "d$k", minLines = 2) { onChange(a.copy(description = it)) }
        Spacer(Modifier.height(6.dp))
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Muted("بُعد زندگی:", Modifier.align(Alignment.CenterVertically))
            Dropdown(dims.find { it.id == a.dimensionId }?.name ?: "—",
                listOf<Pair<String?, String>>(null to "—") + dims.filter { it.active || it.id == a.dimensionId }
                    .map { it.id to (it.name + if (it.kind == "category") " (دسته جدا)" else "") }) { onChange(a.copy(dimensionId = it)) }
        }
        Muted("میزان خروج از ناحیه امنم")
        Segmented(COMFORT_LABELS.mapIndexed { j, l -> (j + 1) to l }, a.comfort) { onChange(a.copy(comfort = it)) }
        if (!plan) {
            Spacer(Modifier.height(6.dp))
            Text("موانع و کشف: ", fontWeight = FontWeight.Bold)
            Muted(REFLECTION_Q)
            BoundField(a.reflection, null, "r$k", minLines = 3) { onChange(a.copy(reflection = it)) }
        }
        Spacer(Modifier.height(4.dp))
        BoundField(a.need, "نیاز", "n$k") { onChange(a.copy(need = it)) }
        if (a.need.isBlank()) Chips(sugg.need) { onChange(a.copy(need = it)) }
        BoundField(a.value, "ارزش", "v$k") { onChange(a.copy(value = it)) }
        if (a.value.isBlank()) Chips(sugg.value) { onChange(a.copy(value = it)) }
        BoundField(a.ability, "توانایی", "a$k") { onChange(a.copy(ability = it)) }
        if (a.ability.isBlank()) Chips(sugg.ability) { onChange(a.copy(ability = it)) }
        if (!plan) {
            HorizontalDivider(Modifier.padding(vertical = 8.dp), color = K.line)
            Text("امتیاز (${Fmt.num(0)} تا ${Fmt.num(max)}):", fontWeight = FontWeight.Bold)
            Segmented((0..max).map { it to Fmt.num(it) }, a.score, tag = "score$i") { onChange(a.copy(score = it)) }
            Spacer(Modifier.height(6.dp))
            StatusPicker(a.status) { onChange(a.copy(status = it)) }
        }
    }
}

@Composable
fun StatusPicker(value: String, onChange: (String) -> Unit) =
    Segmented(STATUS_LABELS.entries.map { it.key to it.value }, value, clearable = false) { onChange(it ?: "todo") }

@Composable
fun Commitments(day: Day, s: Settings, prev: Day?, compact: Boolean, onChange: (String, CommitmentAnswer) -> Unit) {
    Row(Modifier.fillMaxWidth()) {
        Spacer(Modifier.weight(1f))
        Text("بله", Modifier.width(64.dp), fontWeight = FontWeight.Bold, fontSize = 13.sp)
        Text("خیر", Modifier.width(76.dp), fontWeight = FontWeight.Bold, fontSize = 13.sp)
    }
    day.rules.defs.forEach { def ->
        val a = day.commitments[def.id] ?: CommitmentAnswer()
        val pa = prev?.commitments?.get(def.id)
        fun answer(ans: String) {
            val newAns = if (a.answer == ans) null else ans
            var na = a.copy(answer = newAns)
            if (def.kind == "multiplier" && newAns == "no" && a.n == null) na = na.copy(n = 1)
            if (def.kind == "streak" && newAns == "yes") {
                if (a.streak == null) na = na.copy(streak = Logic.suggestStreak(pa))
                if (a.behavior.isNullOrBlank() && !pa?.behavior.isNullOrBlank()) na = na.copy(behavior = pa?.behavior)
            }
            onChange(def.id, na)
        }
        Row(Modifier.fillMaxWidth().padding(vertical = 4.dp), verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text(Export.labelOf(def.label, s), fontSize = 14.sp)
                if (def.kind == "star") TextButton({ Nav.go(Screen.StarList) }, contentPadding = PaddingValues(0.dp)) { Text("باز کردن ستاره ←", fontSize = 12.sp) }
            }
            Spacer(Modifier.width(4.dp))
            Pill(Fmt.signed(def.yesPts), a.answer == "yes", Modifier.width(60.dp).testTag("yes-${def.id}"), onColor = K.good) { answer("yes") }
            Spacer(Modifier.width(4.dp))
            val noLabel = if (def.kind == "multiplier") "${Fmt.num(def.noPts)}×${if (a.answer == "no") Fmt.num(a.n ?: 1) else "…"}" else Fmt.num(def.noPts)
            Pill(noLabel, a.answer == "no", Modifier.width(72.dp).testTag("no-${def.id}"), onColor = K.bad) { answer("no") }
        }
        if (def.kind == "multiplier" && a.answer == "no") {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                IntField(a.n ?: 1, "n${day.date}", 1, 99, label = "تعداد") { onChange(def.id, a.copy(n = it ?: 1)) }
                BoundField(a.note ?: "", "چه چیزی را شکستم؟", "note${day.date}", Modifier.weight(1f)) { onChange(def.id, a.copy(note = it)) }
            }
        }
        if (def.kind == "streak" && !compact) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                BoundField(a.behavior ?: "", "رفتار", "b${def.id}${day.date}", Modifier.weight(1f)) { onChange(def.id, a.copy(behavior = it)) }
                Text("برای")
                IntField(a.streak, "s${def.id}${day.date}${a.answer}", 0, 9999) { onChange(def.id, a.copy(streak = it)) }
                Text("روز متوالی")
            }
        }
        HorizontalDivider(color = Color(0xFFEFEDE8))
    }
}
