@file:OptIn(ExperimentalLayoutApi::class)

package app.kimiagar.ui

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import app.kimiagar.core.*

@Composable
fun KCard(modifier: Modifier = Modifier, border: Color = K.line, content: @Composable ColumnScope.() -> Unit) {
    Column(
        modifier.fillMaxWidth().padding(vertical = 6.dp).clip(RoundedCornerShape(14.dp)).background(K.card)
            .border(1.dp, border, RoundedCornerShape(14.dp)).padding(12.dp),
        content = content,
    )
}

@Composable
fun Banner(text: String, warn: Boolean = false, onClick: (() -> Unit)? = null) {
    Text(
        text,
        Modifier.fillMaxWidth().padding(vertical = 4.dp).clip(RoundedCornerShape(10.dp))
            .background(if (warn) K.warnSoft else K.accentSoft)
            .then(if (onClick != null) Modifier.clickable(onClick = onClick) else Modifier).padding(10.dp),
    )
}

@Composable
fun Muted(text: String, modifier: Modifier = Modifier) = Text(text, modifier, color = K.muted, style = MaterialTheme.typography.bodySmall)

@Composable
fun Title(text: String) = Text(text, style = MaterialTheme.typography.titleMedium, modifier = Modifier.padding(bottom = 4.dp))

/** One pill/segment. */
@Composable
fun Pill(text: String, on: Boolean, modifier: Modifier = Modifier, onColor: Color = K.accent, onClick: () -> Unit) {
    Box(
        modifier.clip(RoundedCornerShape(8.dp)).background(if (on) onColor else Color.White)
            .border(1.dp, if (on) onColor else K.line, RoundedCornerShape(8.dp))
            .clickable(onClick = onClick).defaultMinSize(minWidth = 36.dp, minHeight = 36.dp).padding(horizontal = 10.dp, vertical = 6.dp),
        contentAlignment = Alignment.Center,
    ) { Text(text, color = if (on) Color.White else K.ink, style = MaterialTheme.typography.bodyMedium) }
}

/** Segmented choice; tapping the selected one again clears it (when clearable). */
@Composable
fun <T> Segmented(options: List<Pair<T, String>>, value: T?, clearable: Boolean = true, tag: String? = null, onChange: (T?) -> Unit) {
    FlowRow(horizontalArrangement = Arrangement.spacedBy(4.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
        options.forEach { (v, label) ->
            Pill(label, value == v, if (tag != null) Modifier.testTag("$tag-$v") else Modifier) { onChange(if (value == v && clearable) null else v) }
        }
    }
}

@Composable
fun Chips(items: List<String>, onPick: (String) -> Unit) {
    if (items.isEmpty()) return
    FlowRow(Modifier.padding(top = 2.dp), horizontalArrangement = Arrangement.spacedBy(4.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
        items.forEach { c ->
            Text(c, Modifier.clip(RoundedCornerShape(50)).background(K.accentSoft).clickable { onPick(c) }.padding(horizontal = 10.dp, vertical = 3.dp),
                color = K.accent, style = MaterialTheme.typography.bodySmall)
        }
    }
}

/** Text field keeping its own state (smooth typing), pushing every change out. Re-syncs when `key` changes. */
@Composable
fun BoundField(value: String, label: String?, key: Any?, modifier: Modifier = Modifier, minLines: Int = 1, onChange: (String) -> Unit) {
    var t by remember(key) { mutableStateOf(value) }
    LaunchedEffect(value) { if (value != t) t = value }
    OutlinedTextField(
        t, { t = it; onChange(it) }, modifier.fillMaxWidth(),
        label = label?.let { { Text(it) } }, minLines = minLines,
        textStyle = MaterialTheme.typography.bodyMedium,
    )
}

/** Integer field that accepts Persian digits; invalid input is shown red and not saved. */
@Composable
fun IntField(value: Int?, key: Any?, min: Int, max: Int, width: Dp = 72.dp, label: String? = null, onChange: (Int?) -> Unit) {
    var t by remember(key) { mutableStateOf(value?.let { Fmt.num(it) } ?: "") }
    LaunchedEffect(value) { if (Fmt.parseInt(t) != value) t = value?.let { Fmt.num(it) } ?: "" }
    val v = Fmt.parseInt(t)
    val bad = t.isNotBlank() && (v == null || v < min || v > max)
    OutlinedTextField(
        t,
        { s -> t = s; if (s.isBlank()) onChange(null) else Fmt.parseInt(s)?.takeIf { it in min..max }?.let(onChange) },
        Modifier.width(width), isError = bad, singleLine = true, label = label?.let { { Text(it) } },
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
        textStyle = MaterialTheme.typography.bodyMedium.copy(textAlign = TextAlign.Center),
    )
}

@Composable
fun <T> Dropdown(current: String, items: List<Pair<T, String>>, onPick: (T) -> Unit) {
    var open by remember { mutableStateOf(false) }
    Box {
        OutlinedButton({ open = true }, shape = RoundedCornerShape(8.dp)) { Text(current + "  ▾", color = K.ink) }
        DropdownMenu(open, { open = false }) {
            items.forEach { (v, l) -> DropdownMenuItem(text = { Text(l) }, onClick = { open = false; onPick(v) }) }
        }
    }
}

/** The five workbook faces: 1 big smile … 5 shocked/distressed. */
@Composable
fun Face(mood: Int, size: Dp = 30.dp, color: Color = K.ink) {
    Canvas(Modifier.size(size)) {
        val w = this.size.width
        val sw = w * 0.055f
        drawCircle(color, w * 0.44f, style = Stroke(sw))
        drawCircle(color, w * 0.055f, Offset(w * 0.36f, w * 0.39f))
        drawCircle(color, w * 0.055f, Offset(w * 0.64f, w * 0.39f))
        when (mood) {
            1 -> drawArc(color, 0f, 180f, true, Offset(w * 0.30f, w * 0.48f), Size(w * 0.40f, w * 0.30f))
            2 -> drawArc(color, 20f, 140f, false, Offset(w * 0.30f, w * 0.44f), Size(w * 0.40f, w * 0.28f), style = Stroke(sw, cap = StrokeCap.Round))
            3 -> drawLine(color, Offset(w * 0.34f, w * 0.66f), Offset(w * 0.66f, w * 0.66f), sw, StrokeCap.Round)
            4 -> drawArc(color, 200f, 140f, false, Offset(w * 0.30f, w * 0.60f), Size(w * 0.40f, w * 0.28f), style = Stroke(sw, cap = StrokeCap.Round))
            else -> drawOval(color, Offset(w * 0.41f, w * 0.58f), Size(w * 0.18f, w * 0.22f))
        }
    }
}

@Composable
fun MoodPicker(value: Int?, onChange: (Int?) -> Unit) {
    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        (1..5).forEach { m ->
            val on = value == m
            Box(
                Modifier.size(48.dp).clip(CircleShape).background(if (on) K.accentSoft else Color.White)
                    .border(1.dp, if (on) K.accent else K.line, CircleShape).clickable { onChange(if (on) null else m) },
                contentAlignment = Alignment.Center,
            ) { Face(m, 32.dp, if (on) K.accent else K.muted) }
        }
    }
    value?.let { Muted(MOOD_TITLES[it - 1]) }
}

/** Vertical bars; first item is drawn on the right (RTL time flow). */
@Composable
fun BarChart(values: List<Int?>, labels: List<String>, max: Int = 100, onClick: ((Int) -> Unit)? = null) {
    Column {
        Row(Modifier.fillMaxWidth().height(110.dp), verticalAlignment = Alignment.Bottom) {
            values.forEachIndexed { i, v ->
                Box(Modifier.weight(1f).fillMaxHeight().then(if (onClick != null) Modifier.clickable { onClick(i) } else Modifier), contentAlignment = Alignment.BottomCenter) {
                    if (v != null) {
                        val f = (v.coerceIn(0, max).toFloat() / max).coerceAtLeast(0.01f)
                        Box(Modifier.fillMaxWidth(0.62f).fillMaxHeight(f).clip(RoundedCornerShape(topStart = 3.dp, topEnd = 3.dp)).background(K.accent))
                    }
                }
            }
        }
        HorizontalDivider(color = K.line)
        Row(Modifier.fillMaxWidth()) {
            values.forEachIndexed { i, v ->
                Column(Modifier.weight(1f), horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(v?.let { Fmt.num(it) } ?: "", fontSize = 10.sp, fontWeight = FontWeight.Bold)
                    Text(labels[i], fontSize = 10.sp, color = K.muted)
                }
            }
        }
    }
}

@Composable
fun HBar(name: String, value: Int, max: Int) {
    Row(Modifier.fillMaxWidth().padding(vertical = 2.dp), verticalAlignment = Alignment.CenterVertically) {
        Text(name, Modifier.width(130.dp), style = MaterialTheme.typography.bodySmall)
        Box(Modifier.weight(1f).height(12.dp)) {
            if (value > 0) Box(Modifier.fillMaxWidth(value.toFloat() / maxOf(1, max)).fillMaxHeight().clip(RoundedCornerShape(3.dp)).background(K.accent))
        }
        Text(Fmt.num(value), Modifier.width(30.dp), textAlign = TextAlign.Center, fontWeight = FontWeight.Bold)
    }
}

/** Jalali month grid, Saturday first. */
@Composable
fun MonthGrid(jy: Int, jm: Int, cell: @Composable RowScope.(iso: String) -> Unit) {
    val first = Jalali.toIso(jy, jm, 1)!!
    val lead = Dates.satIndex(first)
    val len = Jalali.monthLength(jy, jm)
    val cells: List<String?> = List(lead) { null } + (0 until len).map { Dates.add(first, it.toLong()) }
    Column(verticalArrangement = Arrangement.spacedBy(3.dp)) {
        Row { listOf("ش", "ی", "د", "س", "چ", "پ", "ج").forEach { Text(it, Modifier.weight(1f), textAlign = TextAlign.Center, color = K.muted, fontSize = 12.sp) } }
        cells.chunked(7).forEach { week ->
            Row(horizontalArrangement = Arrangement.spacedBy(3.dp)) {
                week.forEach { iso -> if (iso == null) Spacer(Modifier.weight(1f)) else cell(iso) }
                repeat(7 - week.size) { Spacer(Modifier.weight(1f)) }
            }
        }
    }
}

@Composable
fun MonthNav(jy: Int, jm: Int, onChange: (Int, Int) -> Unit) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
        TextButton({ if (jm == 1) onChange(jy - 1, 12) else onChange(jy, jm - 1) }) { Text("→", fontSize = 20.sp) }
        Text("${Jalali.MONTHS[jm - 1]} ${Fmt.num(jy)}", fontWeight = FontWeight.Bold)
        TextButton({ if (jm == 12) onChange(jy + 1, 1) else onChange(jy, jm + 1) }) { Text("←", fontSize = 20.sp) }
    }
}

/** Button showing a Jalali date; opens a month picker. Emits Gregorian ISO (or null when cleared). */
@Composable
fun JalaliDateButton(value: String?, allowClear: Boolean = true, placeholder: String = "انتخاب تاریخ", onChange: (String?) -> Unit) {
    var open by remember { mutableStateOf(false) }
    OutlinedButton({ open = true }, shape = RoundedCornerShape(8.dp)) { Text(value?.let { Fmt.jalali(it) } ?: placeholder, color = K.ink) }
    if (open) {
        val start = Jalali.fromIso(value ?: Dates.today())
        var ym by remember { mutableStateOf(start.jy to start.jm) }
        AlertDialog(
            onDismissRequest = { open = false },
            confirmButton = { if (allowClear && value != null) TextButton({ onChange(null); open = false }) { Text("پاک کردن") } },
            dismissButton = { TextButton({ open = false }) { Text("بستن") } },
            text = {
                Column {
                    MonthNav(ym.first, ym.second) { y, m -> ym = y to m }
                    MonthGrid(ym.first, ym.second) { iso ->
                        val sel = iso == value
                        Box(
                            Modifier.weight(1f).height(38.dp).clip(RoundedCornerShape(6.dp)).background(if (sel) K.accent else Color.White)
                                .border(1.dp, if (iso == Dates.today()) K.accent else K.line, RoundedCornerShape(6.dp))
                                .clickable { onChange(iso); open = false },
                            contentAlignment = Alignment.Center,
                        ) { Text(Fmt.num(Jalali.fromIso(iso).jd), color = if (sel) Color.White else K.ink) }
                    }
                }
            },
        )
    }
}

@Composable
fun ScorePanel(day: Day) {
    val s = Scoring.scoreDay(day)
    Row(
        Modifier.fillMaxWidth().padding(vertical = 6.dp).clip(RoundedCornerShape(14.dp)).background(K.accent).padding(14.dp),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Row(verticalAlignment = Alignment.Bottom) {
            Text("امتیاز امروز  ", color = Color.White)
            Text(Fmt.num(s.total), color = if (s.total < 0) Color(0xFFFFD2CC) else Color.White, fontSize = 34.sp, fontWeight = FontWeight.Bold)
        }
        Column(horizontalAlignment = Alignment.End) {
            Text("اقدام‌ها ${Fmt.num(s.actions)} · تعهدات ${Fmt.num(s.commitments)}", color = Color.White, fontSize = 13.sp)
            Text("کسر از ${Fmt.num(s.max)}: ${Fmt.num(-s.lost)}", color = Color.White.copy(alpha = 0.85f), fontSize = 13.sp)
        }
    }
}

@Composable
fun PrimaryButton(text: String, modifier: Modifier = Modifier, enabled: Boolean = true, onClick: () -> Unit) =
    Button(onClick, modifier, enabled = enabled, shape = RoundedCornerShape(10.dp)) { Text(text, color = Color.White) }

@Composable
fun GhostButton(text: String, modifier: Modifier = Modifier, onClick: () -> Unit) =
    OutlinedButton(onClick, modifier, shape = RoundedCornerShape(10.dp), border = BorderStroke(1.dp, K.accent)) { Text(text, color = K.accent) }
