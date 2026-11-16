@file:OptIn(ExperimentalLayoutApi::class)

package app.kimiagar.ui

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import app.kimiagar.core.*
import app.kimiagar.data.Repo

@Composable
fun TodayScreen() {
    val data by Repo.data.collectAsState()
    val s = data.s
    val t = today()
    var day by remember(t) { mutableStateOf(Repo.dayOrBlank(t)) }
    LaunchedEffect(data.days) { Repo.day(t)?.let { if (it != day) day = it } } // reflect changes made elsewhere (e.g. Star)
    fun set(d: Day) { day = d; Repo.putDay(d) }
    val prev = remember(t) { Repo.prevDay(t) }

    Text(Fmt.jalaliLong(t), style = MaterialTheme.typography.titleLarge)
    day.dayNumber?.let { Muted("روز ${Fmt.num(it)} از دوره ${Fmt.num(s.courseLength)} روزه") }

    if (Logic.backupDue(s.lastBackupAt, System.currentTimeMillis(), data.days.isNotEmpty()))
        Banner("یک هفته از آخرین پشتیبان گذشته است. برای پشتیبان‌گیری اینجا بزنید.", warn = true) { Nav.tab(Screen.Settings) }
    Logic.nextExperienceDue(data.experiences, s.courseStart, s.expIntervalDays)?.let { due ->
        val left = Logic.daysUntil(due, t)
        val msg = when { left > 0 -> "${Fmt.num(left)} روز مانده (${Fmt.jalali(due)})"; left == 0L -> "موعدش امروز است"; else -> "${Fmt.num(-left)} روز از موعد گذشته" }
        Banner("تجربهٔ جدید: $msg", warn = left < 0) { Nav.tab(Screen.Experiences) }
    }

    KCard {
        Text("قصد امروز جاری کردن: " + day.intention.ifBlank { "— (شب قبل در «برنامه‌ریزی فردا» بنویسید)" }, fontWeight = FontWeight.Bold)
        day.actions.forEachIndexed { i, a ->
            HorizontalDivider(Modifier.padding(vertical = 6.dp), color = K.line)
            Text("${Fmt.num(i + 1)}. " + a.topic.ifBlank { "اقدام ${Fmt.num(i + 1)}" } + (if (a.description.isNotBlank()) " — ${a.description}" else ""))
            StatusPicker(a.status) { st -> set(day.copy(actions = day.actions.mapIndexed { j, x -> if (j == i) x.copy(status = st) else x })) }
        }
        Spacer(Modifier.height(8.dp))
        FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            PrimaryButton("صفحهٔ کامل امروز (امتیاز و بازتاب)") { Nav.go(Screen.DayPage(t)) }
            GhostButton("برنامه‌ریزی فردا") { Nav.go(Screen.DayPage(Dates.add(t, 1), plan = true)) }
        }
    }

    KCard {
        Title("تعهدات روزانه")
        Commitments(day, s, prev, compact = true) { id, ca -> set(day.copy(commitments = day.commitments + (id to ca))) }
    }
    ScorePanel(day)

    val chrono = (0..13).map { Dates.add(t, (it - 13).toLong()) }
    val byDate = data.days.associateBy { it.date }
    KCard {
        val w = Logic.weekAverage(data.days, t)
        Title("۱۴ روز اخیر" + (w?.let { " · میانگین این هفته ${Fmt.num(it)}" } ?: ""))
        BarChart(
            chrono.map { iso -> byDate[iso]?.takeIf { Logic.isScored(it) }?.let { Logic.total(it) } },
            chrono.map { Fmt.num(Jalali.fromIso(it).jd) },
        ) { i -> Nav.go(Screen.DayPage(chrono[i])) }
    }

    val drafts = data.starExercises.count { it.status == "draft" }
    val lastDone = data.starExercises.filter { it.status == "done" }.maxByOrNull { it.doneAt ?: 0 }
    KCard {
        Title("ستاره بازنویسی")
        if (drafts > 0) Text("${Fmt.num(drafts)} تمرین نیمه‌کاره")
        lastDone?.let { Muted("آخرین تکمیل: ${Fmt.jalali(it.date)}") }
        if (drafts == 0 && lastDone == null) Muted("هنوز تمرینی ثبت نشده.")
        GhostButton(if (drafts > 0) "ادامه" else "شروع") { Nav.tab(Screen.StarList) }
    }

    val since = Dates.add(t, -13)
    val counts = data.days.filter { it.date >= since }.flatMap { it.actions }.mapNotNull { it.dimensionId }.groupingBy { it }.eachCount()
    KCard {
        Title("تعادل (۱۴ روز)")
        if (counts.isEmpty()) Muted("اقدامی به ابعاد زندگی وصل نشده.")
        counts.entries.sortedByDescending { it.value }.take(5).forEach { (id, n) ->
            Text("${data.dimensions.find { it.id == id }?.name ?: "?"}: ${Fmt.num(n)}")
        }
        GhostButton("جزئیات") { Nav.tab(Screen.Balance) }
    }
}
