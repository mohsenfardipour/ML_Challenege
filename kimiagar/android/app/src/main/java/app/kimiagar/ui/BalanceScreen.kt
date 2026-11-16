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
fun BalanceScreen() {
    val data by Repo.data.collectAsState()
    val t = today()
    var win by remember { mutableStateOf(14) }
    var dimId by remember { mutableStateOf<String?>(null) }
    var rating by remember { mutableStateOf<Int?>(null) }
    var note by remember { mutableStateOf("") }
    var formKey by remember { mutableStateOf(0) }
    val since = Dates.add(t, -(win - 1).toLong())
    val counts = data.days.filter { it.date in since..t }.flatMap { it.actions }.mapNotNull { it.dimensionId }.groupingBy { it }.eachCount()
    val life = data.dimensions.filter { it.kind == "dimension" && it.active }.sortedBy { it.order }
    val cats = data.dimensions.filter { it.kind == "category" && it.active }
    val latest = data.dimRatings.sortedBy { it.date }.associateBy { it.dimensionId }

    Text("تعادل زندگی", style = MaterialTheme.typography.titleLarge)
    Muted("برای آگاهی است، نه تکلیف روزانه.")
    KCard {
        Title("تعداد اقدام‌ها در هر بُعد")
        Segmented(listOf(14 to "${Fmt.num(14)} روز", 30 to "${Fmt.num(30)} روز"), win, clearable = false) { if (it != null) win = it }
        Spacer(Modifier.height(6.dp))
        val max = (life + cats).maxOfOrNull { counts[it.id] ?: 0 } ?: 1
        life.forEach { HBar(it.name, counts[it.id] ?: 0, max) }
        if (cats.isNotEmpty()) {
            Spacer(Modifier.height(6.dp)); Muted("دسته‌های جدا")
            cats.forEach { HBar(it.name, counts[it.id] ?: 0, max) }
        }
    }
    KCard {
        Title("رضایت از هر بُعد (۱ تا ۱۰)")
        life.forEach { d ->
            val r = latest[d.id]
            Row(Modifier.fillMaxWidth().padding(vertical = 3.dp)) {
                Text(d.name, Modifier.weight(1f))
                if (r != null) {
                    Text(Fmt.num(r.rating), Modifier.width(30.dp), fontWeight = FontWeight.Bold)
                    Muted(Fmt.jalali(r.date) + if (r.note.isNotBlank()) " · ${r.note}" else "")
                } else Muted("—")
            }
        }
    }
    KCard {
        Title("ثبت رضایت")
        Dropdown(life.find { it.id == dimId }?.name ?: "انتخاب بُعد…", life.map { it.id to it.name }) { dimId = it }
        Spacer(Modifier.height(4.dp))
        Segmented((1..10).map { it to Fmt.num(it) }, rating) { rating = it }
        BoundField(note, "یادداشت (اختیاری)", "note$formKey") { note = it }
        PrimaryButton("ثبت", enabled = dimId != null && rating != null) {
            Repo.update { d -> d.copy(dimRatings = d.dimRatings + DimRating(Repo.uid(), dimId!!, t, rating!!, note)) }
            rating = null; note = ""; formKey++
        }
    }
}
