package app.kimiagar.data

import android.content.Context
import android.util.AtomicFile
import app.kimiagar.core.*
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.io.File
import java.util.UUID

/**
 * All app data lives in one JSON document (same shape as the backup file) inside the app's private storage.
 * Writes go through AtomicFile, so a crash mid-write never corrupts the data. Changes are kept in memory
 * immediately and written to disk 400 ms after the last change, and always when the app goes to background.
 */
object Repo {
    private lateinit var file: AtomicFile
    private val _data = MutableStateFlow(AppData())
    val data: StateFlow<AppData> get() = _data
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var saveJob: Job? = null
    private val lock = Any()
    var loadError: String? = null
        private set

    fun init(ctx: Context) {
        if (::file.isInitialized) return
        file = AtomicFile(File(ctx.filesDir, "kimiagar.json"))
        _data.value = try {
            if (file.baseFile.exists()) KJson.decodeFromString(AppData.serializer(), String(file.readFully(), Charsets.UTF_8)) else AppData()
        } catch (e: Exception) {
            // Never overwrite a file we could not read: keep a copy and start empty.
            loadError = "فایل اطلاعات خوانده نشد؛ یک نسخه از آن کنار گذاشته شد."
            file.baseFile.copyTo(File(ctx.filesDir, "kimiagar-unreadable-${System.currentTimeMillis()}.json"), overwrite = true)
            AppData()
        }
        Fmt.persianDigits = _data.value.s.digits != "latin"
    }

    fun update(fn: (AppData) -> AppData) {
        synchronized(lock) { _data.value = fn(_data.value) }
        Fmt.persianDigits = _data.value.s.digits != "latin"
        saveJob?.cancel()
        saveJob = scope.launch { delay(400); flush() }
    }

    fun flush() = synchronized(lock) {
        val bytes = KJson.encodeToString(AppData.serializer(), _data.value).toByteArray(Charsets.UTF_8)
        val out = file.startWrite()
        try { out.write(bytes); file.finishWrite(out) } catch (e: Exception) { file.failWrite(out); throw e }
    }

    fun replaceAll(d: AppData) { update { d }; saveJob?.cancel(); flush() }

    fun uid(): String = UUID.randomUUID().toString()

    // ---- helpers ----
    fun settings(fn: (Settings) -> Settings) = update { it.copy(settings = listOf(fn(it.s))) }

    fun day(date: String): Day? = _data.value.days.find { it.date == date }

    /** A blank day (not saved until edited) with today's rule snapshot and the suggested page number. */
    fun dayOrBlank(date: String): Day = day(date) ?: _data.value.let { d ->
        Day(date = date, dayNumber = Logic.suggestDayNumber(d.days, date), rules = buildRules(d.s.cardMax, d.commitmentDefs))
    }

    fun putDay(day: Day) = update { d ->
        val nd = day.copy(updatedAt = System.currentTimeMillis())
        d.copy(days = (d.days.filter { it.date != day.date } + nd).sortedBy { it.date })
    }

    fun prevDay(date: String): Day? = _data.value.days.filter { it.date < date }.maxByOrNull { it.date }
}
