@file:OptIn(ExperimentalLayoutApi::class)

package app.kimiagar.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import app.kimiagar.core.*
import app.kimiagar.data.Repo

@Composable
fun HistoryScreen() {
    val data by Repo.data.collectAsState()
    val t = today()
    val tj = Jalali.fromIso(t)
    var ym by remember { mutableStateOf(tj.jy to tj.jm) }
    var from by remember { mutableStateOf<String?>(null) }
    var to by remember { mutableStateOf<String?>(null) }
    var dim by remember { mutableStateOf<String?>(null) }
    val byDate = data.days.associateBy { it.date }

    Text("تاریخچه", style = MaterialTheme.typography.titleLarge)
    KCard {
        MonthNav(ym.first, ym.second) { y, m -> ym = y to m }
        MonthGrid(ym.first, ym.second) { iso ->
            val d = byDate[iso]
            val bg = when { d == null -> Color.White; Logic.isScored(d) -> K.goodSoft; Logic.isPlanned(d) -> K.accentSoft; else -> Color.White }
            Column(
                Modifier.weight(1f).heightIn(min = 54.dp).clip(RoundedCornerShape(8.dp)).background(bg)
                    .border(if (iso == t) 2.dp else 1.dp, if (iso == t) K.accent else K.line, RoundedCornerShape(8.dp))
                    .clickable { Nav.go(Screen.DayPage(iso)) }.padding(2.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                Text(Fmt.num(Jalali.fromIso(iso).jd), color = K.muted, fontSize = 11.sp)
                if (d != null && Logic.isScored(d)) Text(Fmt.num(Logic.total(d)), fontWeight = FontWeight.Bold, fontSize = 13.sp)
                d?.dayNumber?.let { Text("روز ${Fmt.num(it)}", fontSize = 9.sp, color = K.muted) }
            }
        }
        Spacer(Modifier.height(6.dp))
        val first = Jalali.toIso(ym.first, ym.second, 1)!!
        Text("میانگین این هفته: ${Logic.weekAverage(data.days, t)?.let { Fmt.num(it) } ?: "—"}   ·   میانگین این ماه: ${Logic.monthAverage(data.days, first)?.let { Fmt.num(it) } ?: "—"}")
    }

    val list = data.days.filter { (from == null || it.date >= from!!) && (to == null || it.date <= to!!) }
        .filter { d -> dim == null || d.actions.any { it.dimensionId == dim } }.asReversed()
    KCard {
        FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text("از", Modifier.align(Alignment.CenterVertically)); JalaliDateButton(from) { from = it }
            Text("تا", Modifier.align(Alignment.CenterVertically)); JalaliDateButton(to) { to = it }
            Dropdown(data.dimensions.find { it.id == dim }?.name ?: "همهٔ ابعاد",
                listOf<Pair<String?, String>>(null to "همهٔ ابعاد") + data.dimensions.map { it.id to it.name }) { dim = it }
        }
        val avg = Logic.averageBetween(list, "0000-01-01", "9999-12-31")
        Text("میانگین بازهٔ انتخاب‌شده: ${avg?.let { Fmt.num(it) } ?: "—"} (${Fmt.num(list.count { Logic.isScored(it) })} روز امتیازدار)")
        if (list.isEmpty()) Muted("روزی ثبت نشده.")
        list.forEach { d ->
            HorizontalDivider(color = K.line)
            Row(Modifier.fillMaxWidth().clickable { Nav.go(Screen.DayPage(d.date)) }.padding(vertical = 8.dp), verticalAlignment = Alignment.CenterVertically) {
                Text(d.dayNumber?.let { "روز ${Fmt.num(it)}" } ?: "", Modifier.width(56.dp), fontSize = 13.sp)
                Text(Fmt.jalali(d.date), Modifier.width(92.dp), fontSize = 13.sp)
                val st = when { Logic.isScored(d) -> "بررسی شد" to K.good; Logic.isPlanned(d) -> "برنامه‌ریزی شد" to K.accent; else -> "خالی" to K.muted }
                Column(Modifier.weight(1f)) {
                    Text(st.first, color = st.second, fontSize = 12.sp)
                    if (d.intention.isNotBlank()) Muted(d.intention)
                }
                d.mood?.let { Face(it, 22.dp) }
                Text(if (Logic.isScored(d)) Fmt.num(Logic.total(d)) else "—", Modifier.width(40.dp), fontWeight = FontWeight.Bold)
            }
        }
    }
}
