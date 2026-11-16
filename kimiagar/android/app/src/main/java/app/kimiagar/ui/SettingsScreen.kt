@file:OptIn(ExperimentalLayoutApi::class)

package app.kimiagar.ui

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import app.kimiagar.core.*
import app.kimiagar.data.Repo
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

private val KIND_LABEL = linkedMapOf("simple" to "ساده", "multiplier" to "ضرب در تعداد (خیر)", "streak" to "روزهای متوالی", "star" to "ستاره بازنویسی")

@Composable
fun SettingsScreen() {
    val data by Repo.data.collectAsState()
    val s = data.s
    Text("تنظیمات", style = MaterialTheme.typography.titleLarge)
    KCard {
        BoundField(s.name, "نام کیمیاگر", "name") { v -> Repo.settings { it.copy(name = v) } }
        FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text("تاریخ شروع دوره:", Modifier.align(Alignment.CenterVertically))
            JalaliDateButton(s.courseStart) { v -> Repo.settings { it.copy(courseStart = v) } }
        }
        FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            IntField(s.courseLength, "len", 1, 999, 110.dp, "طول دوره") { v -> v?.let { n -> Repo.settings { it.copy(courseLength = n) } } }
            IntField(s.expIntervalDays, "exp", 1, 365, 130.dp, "فاصلهٔ تجربه") { v -> v?.let { n -> Repo.settings { it.copy(expIntervalDays = n) } } }
        }
        Text("سقف امتیاز کارت‌ها:")
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            s.cardMax.forEachIndexed { i, m ->
                IntField(m, "cm$i", 0, 100) { v -> v?.let { n -> Repo.settings { st -> st.copy(cardMax = st.cardMax.mapIndexed { j, x -> if (j == i) n else x }) } } }
            }
        }
        BoundField(s.walkBlank, "جای خالی ردیف پیاده‌روی («$WALK_BLANK»)", "walk") { v -> Repo.settings { it.copy(walkBlank = v) } }
        Text("ارقام:")
        Segmented(listOf("fa" to "۱۲۳", "latin" to "123"), s.digits, clearable = false) { v -> v?.let { d -> Repo.settings { it.copy(digits = d) } } }
    }
    BackupSection()
    CommitmentEditor(data.commitmentDefs)
    DimensionEditor(data.dimensions)
}

@Composable
private fun BackupSection() {
    val ctx = LocalContext.current
    val data by Repo.data.collectAsState()
    var msg by remember { mutableStateOf<String?>(null) }
    var pending by remember { mutableStateOf<BackupFile?>(null) }
    val stamp = today()

    fun write(uri: Uri?, text: String, ok: String) {
        if (uri == null) return
        msg = try { ctx.contentResolver.openOutputStream(uri, "wt")!!.use { it.write(text.toByteArray(Charsets.UTF_8)) }; ok } catch (e: Exception) { "خطا در ذخیرهٔ فایل: ${e.message}" }
    }
    val backupL = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/json")) { uri ->
        write(uri, Backup.export(Repo.data.value), "فایل پشتیبان ساخته شد. آن را جای امنی نگه دارید.")
        if (uri != null) Repo.settings { it.copy(lastBackupAt = System.currentTimeMillis()) }
    }
    val restoreL = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri == null) return@rememberLauncherForActivityResult
        try {
            val text = ctx.contentResolver.openInputStream(uri)!!.use { String(it.readBytes(), Charsets.UTF_8) }
            pending = Backup.parse(text)
        } catch (e: Exception) { msg = "خطا: ${e.message}" }
    }
    val daysL = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("text/csv")) { write(it, Export.daysCsv(Repo.data.value.days), "CSV ذخیره شد.") }
    val actsL = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("text/csv")) { write(it, Export.actionsCsv(Repo.data.value.days, Repo.data.value.dimensions), "CSV ذخیره شد.") }
    val comL = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("text/csv")) { write(it, Export.commitmentsCsv(Repo.data.value.days), "CSV ذخیره شد.") }

    KCard {
        Title("پشتیبان و خروجی")
        Muted("همهٔ اطلاعات فقط روی همین گوشی است و برنامه اجازهٔ اینترنت ندارد. گوشی و کامپیوتر همگام نمی‌شوند؛ برای انتقال، پشتیبان بگیرید و در دستگاه دیگر بازگردانی کنید (فایل‌ها بین نسخهٔ گوشی و ویندوز یکسان‌اند).")
        data.s.lastBackupAt?.let { Muted("آخرین پشتیبان: " + SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.US).format(Date(it))) }
        FlowRow(Modifier.padding(top = 6.dp), horizontalArrangement = Arrangement.spacedBy(6.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            PrimaryButton("پشتیبان‌گیری (JSON)") { backupL.launch("kimiagar-backup-$stamp.json") }
            GhostButton("بازگردانی از فایل…") { restoreL.launch(arrayOf("application/json", "application/octet-stream", "text/plain", "*/*")) }
        }
        FlowRow(Modifier.padding(top = 6.dp), horizontalArrangement = Arrangement.spacedBy(6.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            GhostButton("CSV روزها") { daysL.launch("kimiagar-days-$stamp.csv") }
            GhostButton("CSV اقدام‌ها") { actsL.launch("kimiagar-actions-$stamp.csv") }
            GhostButton("CSV تعهدات") { comL.launch("kimiagar-commitments-$stamp.csv") }
        }
        msg?.let { Banner(it) }
    }
    pending?.let { b ->
        AlertDialog(
            onDismissRequest = { pending = null },
            title = { Text("بازگردانی پشتیبان") },
            text = {
                Text("همهٔ اطلاعات فعلی با این فایل جایگزین می‌شود.\nروزها: ${Fmt.num(b.data.days.size)} · تمرین ستاره: ${Fmt.num(b.data.starExercises.size)} · تجربه‌ها: ${Fmt.num(b.data.experiences.size)} · رضایت‌ها: ${Fmt.num(b.data.dimRatings.size)}\nتاریخ پشتیبان: ${b.exportedAt}\n\nادامه می‌دهید؟")
            },
            confirmButton = { TextButton({ Repo.replaceAll(b.data); pending = null; msg = "بازگردانی انجام شد ✓" }) { Text("بله، بازگردانی") } },
            dismissButton = { TextButton({ pending = null }) { Text("انصراف") } },
        )
    }
}

private fun putDef(d: CommitmentDef) = Repo.update { a -> a.copy(commitmentDefs = a.commitmentDefs.map { if (it.id == d.id) d else it }) }
private fun putDim(d: Dimension) = Repo.update { a -> a.copy(dimensions = a.dimensions.map { if (it.id == d.id) d else it }) }

/** Swap the `order` of two neighbours. */
private fun <T> swap(list: List<T>, i: Int, dir: Int, order: (T) -> Int, with: (T, Int) -> T, put: (List<T>) -> Unit) {
    val j = i + dir
    if (j !in list.indices) return
    put(listOf(with(list[i], order(list[j])), with(list[j], order(list[i]))))
}

@Composable
private fun CommitmentEditor(all: List<CommitmentDef>) {
    val defs = all.sortedBy { it.order }
    val max = defs.filter { it.active }.sumOf { maxOf(0, it.yesPts) }
    KCard {
        Title("تعهدات روزانه")
        Muted("تغییرات فقط روی روزهای تازه اعمال می‌شود. هر روز قوانین زمان ساخت خودش را نگه می‌دارد (در صفحهٔ روز «اعمال قوانین فعلی»). جمع «بله»ها: ${Fmt.num(max)}")
        defs.forEachIndexed { i, d ->
            HorizontalDivider(Modifier.padding(vertical = 6.dp), color = K.line)
            BoundField(d.label, "${Fmt.num(i + 1)}." + if (d.active) "" else " (غیرفعال)", "lbl${d.id}") { putDef(d.copy(label = it)) }
            FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                IntField(d.yesPts, "y${d.id}", -100, 100, label = "بله") { v -> v?.let { putDef(d.copy(yesPts = it)) } }
                IntField(d.noPts, "n${d.id}", -100, 100, label = "خیر") { v -> v?.let { putDef(d.copy(noPts = it)) } }
                Dropdown(KIND_LABEL[d.kind] ?: d.kind, KIND_LABEL.entries.map { it.key to it.value }) { putDef(d.copy(kind = it)) }
                Row(verticalAlignment = Alignment.CenterVertically) { Checkbox(d.active, { putDef(d.copy(active = it)) }); Text("فعال") }
                TextButton({ swap(defs, i, -1, { it.order }, { x, o -> x.copy(order = o) }) { ch -> ch.forEach(::putDef) } }) { Text("▲") }
                TextButton({ swap(defs, i, 1, { it.order }, { x, o -> x.copy(order = o) }) { ch -> ch.forEach(::putDef) } }) { Text("▼") }
            }
        }
        GhostButton("افزودن تعهد") {
            Repo.update { a -> a.copy(commitmentDefs = a.commitmentDefs + CommitmentDef(Repo.uid(), (a.commitmentDefs.maxOfOrNull { it.order } ?: 0) + 1, "تعهد جدید", 5, 0, "simple", true)) }
        }
    }
}

@Composable
private fun DimensionEditor(all: List<Dimension>) {
    val dims = all.sortedBy { it.order }
    KCard {
        Title("ابعاد زندگی و دسته‌ها")
        dims.forEachIndexed { i, d ->
            HorizontalDivider(Modifier.padding(vertical = 4.dp), color = K.line)
            BoundField(d.name, null, "dim${d.id}") { putDim(d.copy(name = it)) }
            FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                Dropdown(if (d.kind == "category") "دستهٔ جدا" else "بُعد زندگی", listOf("dimension" to "بُعد زندگی", "category" to "دستهٔ جدا")) { putDim(d.copy(kind = it)) }
                Row(verticalAlignment = Alignment.CenterVertically) { Checkbox(d.active, { putDim(d.copy(active = it)) }); Text("فعال") }
                TextButton({ swap(dims, i, -1, { it.order }, { x, o -> x.copy(order = o) }) { ch -> ch.forEach(::putDim) } }) { Text("▲") }
                TextButton({ swap(dims, i, 1, { it.order }, { x, o -> x.copy(order = o) }) { ch -> ch.forEach(::putDim) } }) { Text("▼") }
            }
        }
        GhostButton("افزودن") {
            Repo.update { a -> a.copy(dimensions = a.dimensions + Dimension(Repo.uid(), "جدید", (a.dimensions.maxOfOrNull { it.order } ?: 0) + 1, true, "dimension")) }
        }
    }
}
