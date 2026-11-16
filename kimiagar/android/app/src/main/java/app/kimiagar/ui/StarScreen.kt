@file:OptIn(ExperimentalLayoutApi::class)

package app.kimiagar.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import app.kimiagar.core.*
import app.kimiagar.data.Repo
import kotlinx.serialization.Serializable
import kotlin.math.cos
import kotlin.math.sin

// ---- config (assets/star.json, shared with the desktop app) ----
@Serializable data class StarField(val id: String, val label: String = "")
@Serializable data class StarStep(
    val id: String, val title: String? = null, val color: String? = null, val red: String? = null, val redDetail: String = "",
    val black: String? = null, val question: String = "", val instruction: String = "", val fields: List<StarField> = emptyList(),
)
@Serializable data class StarCenter(val red: String, val black: String)
@Serializable data class StarConfig(val title: String, val intro: String = "", val center: StarCenter, val start: StarStep, val points: List<StarStep>)

private var cached: StarConfig? = null
@Composable
fun starConfig(): StarConfig {
    val ctx = LocalContext.current
    return cached ?: ctx.assets.open("star.json").bufferedReader(Charsets.UTF_8).use { KJson.decodeFromString(StarConfig.serializer(), it.readText()) }.also { cached = it }
}

fun hexColor(h: String?) = h?.let { Color(android.graphics.Color.parseColor(it)) } ?: K.ink

@Composable
fun StarFigure(cfg: StarConfig, active: Int? = null, size: Dp = 200.dp) {
    Canvas(Modifier.size(size)) {
        val c = this.size.width / 2
        val sc = this.size.width / 200f
        fun p(r: Float, deg: Double): Offset { val a = Math.toRadians(deg); return Offset(c + (r * sc * cos(a)).toFloat(), c + (r * sc * sin(a)).toFloat()) }
        drawCircle(Color(0xFFB59A55), 92f * sc, style = Stroke(1.5f))
        // Canvas x grows to the right regardless of RTL; the diagram's clockwise order is preserved.
        cfg.points.forEachIndexed { k, pt ->
            val deg = -90.0 + 60 * k
            val path = Path().apply {
                moveTo(c, c); val a = p(24f, deg - 30); lineTo(a.x, a.y); val b = p(90f, deg); lineTo(b.x, b.y); val d = p(24f, deg + 30); lineTo(d.x, d.y); close()
            }
            drawPath(path, hexColor(pt.color).copy(alpha = if (active == null || active == k) 1f else 0.18f))
        }
        drawCircle(Color.White.copy(alpha = 0.75f), 26f * sc)
    }
}

@Composable
private fun FieldLabel(step: StarStep, f: StarField) {
    when {
        f.label.isNotBlank() -> Text(f.label)
        f.id == "red" && step.red != null -> Text(step.red, color = K.red)
        f.id == "black" && step.black != null -> Text(step.black, fontWeight = FontWeight.Bold)
    }
}

@Composable
fun StarListScreen() {
    val data by Repo.data.collectAsState()
    val cfg = starConfig()
    var del by remember { mutableStateOf<StarExercise?>(null) }
    Text(cfg.title, style = MaterialTheme.typography.titleLarge)
    Box(Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) { StarFigure(cfg) }
    PrimaryButton("شروع تمرین جدید") {
        val ex = StarExercise(id = Repo.uid(), date = today(), createdAt = System.currentTimeMillis())
        Repo.update { d -> d.copy(starExercises = d.starExercises + ex) }
        Nav.go(Screen.StarRun(ex.id))
    }
    val list = data.starExercises.sortedByDescending { it.createdAt }
    val drafts = list.filter { it.status == "draft" }
    val done = list.filter { it.status == "done" }
    @Composable fun row(e: StarExercise, label: String) {
        Row(Modifier.fillMaxWidth().padding(vertical = 4.dp), verticalAlignment = Alignment.CenterVertically) {
            TextButton({ Nav.go(Screen.StarRun(e.id)) }) { Text(label) }
            Muted(e.answers["current"]?.get("text")?.take(60) ?: "", Modifier.weight(1f))
            TextButton({ del = e }) { Text("حذف", color = K.muted) }
        }
    }
    if (drafts.isNotEmpty()) KCard {
        Title("نیمه‌کاره")
        drafts.forEach { e -> row(e, "${Fmt.jalali(e.date)} · " + if (e.current == 0) cfg.start.title.orEmpty() else "${Fmt.num(e.current)} / ${Fmt.num(cfg.points.size)}") }
    }
    KCard {
        Title("تمرین‌های تکمیل‌شده")
        if (done.isEmpty()) Muted("هنوز تمرینی تکمیل نشده.")
        done.forEach { e -> row(e, Fmt.jalali(e.date)) }
    }
    del?.let { e ->
        AlertDialog(onDismissRequest = { del = null }, title = { Text("این تمرین حذف شود؟") },
            confirmButton = { TextButton({ Repo.update { d -> d.copy(starExercises = d.starExercises.filter { it.id != e.id }) }; del = null }) { Text("حذف") } },
            dismissButton = { TextButton({ del = null }) { Text("انصراف") } })
    }
}

@Composable
fun StarRunScreen(id: String) {
    val cfg = starConfig()
    val steps = listOf(cfg.start) + cfg.points
    val summary = steps.size
    var ex by remember(id) { mutableStateOf(Repo.data.value.starExercises.find { it.id == id }) }
    var msg by remember { mutableStateOf<String?>(null) }
    val e = ex ?: run { Muted("تمرین پیدا نشد."); return }
    fun save(n: StarExercise) { ex = n; Repo.update { d -> d.copy(starExercises = d.starExercises.map { if (it.id == n.id) n else it }) } }
    fun setText(stepId: String, fieldId: String, v: String) =
        save(e.copy(answers = e.answers + (stepId to ((e.answers[stepId] ?: emptyMap()) + (fieldId to v)))))

    if (e.current >= summary) {
        Text("${cfg.title} · ${Fmt.jalali(e.date)}", style = MaterialTheme.typography.titleLarge)
        Box(Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) { StarFigure(cfg, size = 160.dp) }
        steps.forEachIndexed { i, s ->
            KCard {
                if (s.title != null) Text(s.title, fontWeight = FontWeight.Bold)
                else Row { Text(s.red.orEmpty(), color = hexColor(s.color), fontWeight = FontWeight.Bold); Text(" ← "); Text(s.black.orEmpty(), fontWeight = FontWeight.Bold) }
                s.fields.forEach { f -> e.answers[s.id]?.get(f.id)?.takeIf { it.isNotBlank() }?.let { Column { FieldLabel(s, f); Text(it) } } }
                TextButton({ save(e.copy(current = i)) }) { Text("ویرایش") }
            }
        }
        FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            PrimaryButton("ثبت «بله» در تعهد ستاره برای این روز") {
                val day = Repo.dayOrBlank(e.date)
                val def = day.rules.defs.find { it.kind == "star" }
                msg = if (def == null) "در قوانین این روز ردیف «ستاره بازنویسی» فعال نیست." else {
                    Repo.putDay(day.copy(commitments = day.commitments + (def.id to (day.commitments[def.id] ?: CommitmentAnswer()).copy(answer = "yes"))))
                    "برای ${Fmt.jalali(e.date)} «بله» ثبت شد."
                }
            }
            GhostButton("بازگشت به فهرست") { Nav.back() }
        }
        msg?.let { Banner(it) }
        return
    }

    val i = e.current
    val step = steps[i]
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(if (i == 0) step.title.orEmpty() else "${Fmt.num(i)} / ${Fmt.num(cfg.points.size)}", fontWeight = FontWeight.Bold)
        Muted(Fmt.jalali(e.date) + if (e.status == "done") " · تکمیل‌شده" else " · ذخیرهٔ خودکار")
    }
    Box(Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) { StarFigure(cfg, if (i == 0) null else i - 1, 160.dp) }
    Column(Modifier.fillMaxWidth(), horizontalAlignment = Alignment.CenterHorizontally) {
        if (i == 0) {
            Text(cfg.center.red, color = K.red); Text(cfg.center.black, fontWeight = FontWeight.Bold)
        } else {
            Text(step.red.orEmpty(), color = K.red, fontWeight = FontWeight.Bold, fontSize = 17.sp)
            if (step.redDetail.isNotBlank()) Text(step.redDetail, color = K.red, fontSize = 13.sp, textAlign = TextAlign.Center)
            Text("↓", color = K.muted)
            Text(step.black.orEmpty(), fontWeight = FontWeight.Bold, fontSize = 17.sp)
        }
    }
    if (i == 0 && cfg.intro.isNotBlank()) Muted(cfg.intro)
    if (step.instruction.isNotBlank()) Muted(step.instruction)
    if (step.question.isNotBlank()) Text(step.question, fontWeight = FontWeight.Bold)
    step.fields.forEach { f ->
        Spacer(Modifier.height(6.dp))
        FieldLabel(step, f)
        BoundField(e.answers[step.id]?.get(f.id) ?: "", null, "$id-${step.id}-${f.id}", minLines = 5) { setText(step.id, f.id, it) }
    }
    Spacer(Modifier.height(8.dp))
    FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        if (i > 0) GhostButton("قبلی") { save(e.copy(current = i - 1)) }
        if (i < steps.size - 1) PrimaryButton("بعدی") { save(e.copy(current = i + 1)) }
        else PrimaryButton("پایان تمرین") { save(e.copy(current = summary, status = "done", doneAt = e.doneAt ?: System.currentTimeMillis())) }
        TextButton({ Nav.back() }) { Text("ذخیره و خروج") }
    }
}
