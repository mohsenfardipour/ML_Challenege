package app.kimiagar.ui

import android.graphics.Bitmap
import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import app.kimiagar.MainActivity
import app.kimiagar.core.*
import app.kimiagar.data.Repo
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode
import java.io.File

/** End-to-end flows on the real activity (Robolectric). Screenshots go to app/build/shots/. */
@RunWith(RobolectricTestRunner::class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
@Config(sdk = [34], qualifiers = "w390dp-h1400dp-xhdpi")
class FlowTest {
    @get:Rule val rule = createAndroidComposeRule<MainActivity>()

    private fun shot(name: String) {
        rule.waitForIdle()
        val v = rule.activity.window.decorView
        val bmp = Bitmap.createBitmap(v.width, v.height, Bitmap.Config.ARGB_8888)
        rule.runOnUiThread { v.draw(android.graphics.Canvas(bmp)) }
        File("build/shots").apply { mkdirs() }.resolve("$name.png").outputStream().use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
    }
    @org.junit.Before fun start() { rule.runOnUiThread { Nav.tab(Screen.Today) }; rule.waitForIdle() }

    private fun nav(s: Screen) { rule.runOnUiThread { Nav.tab(s) }; rule.waitForIdle() }
    private fun tap(tag: String) = rule.onNodeWithTag(tag).performScrollTo().performClick()

    @Test fun planReviewScoreAndPersist() {
        val t = Dates.today()
        rule.onNodeWithText("تعهدات روزانه").assertExists()
        shot("1-today-empty")

        // Plan tomorrow: review-only fields are hidden
        rule.onAllNodesWithText("برنامه‌ریزی فردا")[0].performScrollTo().performClick()
        rule.onNodeWithText("موانع و کشف: ").assertDoesNotExist()
        rule.onNodeWithText("قصد امروز جاری کردن").performTextInput("آرامش")
        shot("2-plan-tomorrow")
        assertEquals("آرامش", Repo.day(Dates.add(t, 1))?.intention)

        // Review today with example B: 10,10,8,4 + row1 No (n=1, "غیبت") + others Yes = 77
        nav(Screen.DayPage(t))
        listOf(10, 10, 8, 4).forEachIndexed { i, v -> tap("score$i-$v") }
        val defs = Repo.dayOrBlank(t).rules.defs
        assertEquals(12, defs.size)
        tap("no-${defs[0].id}")
        rule.onNodeWithText("چه چیزی را شکستم؟").performScrollTo().performTextInput("غیبت")
        defs.drop(1).forEach { tap("yes-${it.id}") }
        rule.onNodeWithText("۷۷").performScrollTo().assertExists()
        shot("3-day-review-77")

        // Persistence: what is on disk restores to the same day and total
        rule.runOnUiThread { Repo.flush() }
        val file = File(rule.activity.filesDir, "kimiagar.json")
        val disk = KJson.decodeFromString(AppData.serializer(), file.readText())
        val saved = disk.days.first { it.date == t }
        assertEquals(77, Scoring.scoreDay(saved).total)
        assertEquals("غیبت", saved.commitments[defs[0].id]?.note)

        // Today dashboard shows the same total
        nav(Screen.Today)
        assertEquals(2, rule.onAllNodesWithText("۷۷").fetchSemanticsNodes().size) // score panel + today's bar in the chart
        shot("4-today-filled")

        nav(Screen.History); shot("5-history")
        nav(Screen.Balance); shot("6-balance")
        nav(Screen.Settings); shot("7-settings")
    }

    @Test fun starSavesAndResumes() {
        nav(Screen.StarList)
        rule.onNodeWithText("شروع تمرین جدید").performClick()
        rule.onNodeWithText("جایگاه فعلی").assertExists()
        shot("8-star-start")
        rule.onNodeWithText("بعدی").performScrollTo().performClick()
        rule.onNodeWithText("بعدی").performScrollTo().performClick()
        rule.onNodeWithText("۲ / ۶").assertExists()
        shot("9-star-step2")
        val id = Repo.data.value.starExercises.last().id
        assertEquals(2, Repo.data.value.starExercises.last().current)

        // leave and come back: resumes at the same step
        nav(Screen.StarList)
        rule.onNodeWithText("نیمه‌کاره").assertExists()
        nav(Screen.StarRun(id))
        rule.onNodeWithText("۲ / ۶").assertExists()
    }
}
