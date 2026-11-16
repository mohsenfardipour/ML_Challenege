@file:OptIn(ExperimentalLayoutApi::class)

package app.kimiagar.ui

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import app.kimiagar.core.*
import app.kimiagar.data.Repo

private fun putExp(e: Experience) = Repo.update { d -> d.copy(experiences = d.experiences.map { if (it.id == e.id) e else it }) }

@Composable
fun ExperiencesScreen() {
    val data by Repo.data.collectAsState()
    val s = data.s
    val t = today()
    val due = Logic.nextExperienceDue(data.experiences, s.courseStart, s.expIntervalDays)
    Text("تجربهٔ جدید هر ${Fmt.num(s.expIntervalDays)} روز", style = MaterialTheme.typography.titleLarge)
    KCard {
        if (due != null) {
            val left = Logic.daysUntil(due, t)
            Text("موعد تجربهٔ بعدی: ${Fmt.jalali(due)} — " + when { left > 0 -> "${Fmt.num(left)} روز مانده"; left == 0L -> "امروز"; else -> "${Fmt.num(-left)} روز گذشته" })
        } else Muted("برای محاسبهٔ موعد، تاریخ شروع دوره را در تنظیمات وارد کنید.")
        PrimaryButton("افزودن تجربه") {
            Repo.update { d -> d.copy(experiences = d.experiences + Experience(id = Repo.uid(), targetDate = due, createdAt = System.currentTimeMillis())) }
        }
    }
    val open = data.experiences.filter { !it.completed }.sortedByDescending { it.createdAt }
    val done = data.experiences.filter { it.completed }.sortedByDescending { it.completedDate }
    open.forEach { ExpCard(it) }
    if (done.isNotEmpty()) Title("انجام‌شده")
    done.forEach { ExpCard(it) }
}

@Composable
private fun ExpCard(e: Experience) {
    var confirmDel by remember { mutableStateOf(false) }
    KCard(border = if (e.completed) K.good else K.line) {
        BoundField(e.intent, "چه چیزی را می‌خواهم امتحان کنم؟", "i${e.id}") { putExp(e.copy(intent = it)) }
        BoundField(e.why, "چرا؟", "w${e.id}", minLines = 2) { putExp(e.copy(why = it)) }
        FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text("تاریخ هدف:", Modifier.align(Alignment.CenterVertically))
            JalaliDateButton(e.targetDate) { putExp(e.copy(targetDate = it)) }
            Row(verticalAlignment = Alignment.CenterVertically) {
                Checkbox(e.completed, { c -> putExp(e.copy(completed = c, completedDate = if (c) (e.completedDate ?: today()) else null)) })
                Text("انجام شد")
            }
        }
        if (e.completed) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("تاریخ انجام: "); JalaliDateButton(e.completedDate, allowClear = false) { it?.let { d -> putExp(e.copy(completedDate = d)) } }
            }
            BoundField(e.learned, "چه چیزی یاد گرفتم؟", "l${e.id}", minLines = 2) { putExp(e.copy(learned = it)) }
            Text("چقدر معنادار/لذت‌بخش بود؟")
            Segmented((1..5).map { it to Fmt.num(it) }, e.rating) { putExp(e.copy(rating = it)) }
        }
        TextButton({ confirmDel = true }) { Text("حذف", color = K.muted) }
    }
    if (confirmDel) AlertDialog(
        onDismissRequest = { confirmDel = false }, title = { Text("حذف شود؟") },
        confirmButton = { TextButton({ Repo.update { d -> d.copy(experiences = d.experiences.filter { it.id != e.id }) }; confirmDel = false }) { Text("حذف") } },
        dismissButton = { TextButton({ confirmDel = false }) { Text("انصراف") } },
    )
}
