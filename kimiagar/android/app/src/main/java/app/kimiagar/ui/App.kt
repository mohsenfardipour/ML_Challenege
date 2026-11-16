package app.kimiagar.ui

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import app.kimiagar.core.Dates
import app.kimiagar.data.Repo

sealed class Screen(val tab: String) {
    data object Today : Screen("today")
    data class DayPage(val date: String, val plan: Boolean = false) : Screen("today")
    data object History : Screen("history")
    data object StarList : Screen("star")
    data class StarRun(val id: String) : Screen("star")
    data object Experiences : Screen("exp")
    data object Balance : Screen("balance")
    data object Settings : Screen("settings")
}

/** Tiny navigation stack. */
object Nav {
    val stack = mutableStateListOf<Screen>(Screen.Today)
    val current get() = stack.last()
    fun go(s: Screen) { stack.add(s) }
    fun tab(s: Screen) { stack.clear(); stack.add(s) }
    fun back(): Boolean = if (stack.size > 1) { stack.removeAt(stack.lastIndex); true } else false
}

private val TABS = listOf(
    Triple(Screen.Today, "امروز", "☀"),
    Triple(Screen.History, "تاریخچه", "▦"),
    Triple(Screen.StarList, "ستاره", "✶"),
    Triple(Screen.Experiences, "تجربه", "◇"),
    Triple(Screen.Balance, "تعادل", "◎"),
    Triple(Screen.Settings, "تنظیمات", "⚙"),
)

@Composable
fun KimiagarApp() {
    KimiagarTheme {
        val cur = Nav.current
        BackHandler(enabled = Nav.stack.size > 1) { Nav.back() }
        Scaffold(
            containerColor = K.paper,
            bottomBar = {
                NavigationBar(containerColor = K.card, tonalElevation = 0.dp) {
                    TABS.forEach { (s, label, icon) ->
                        NavigationBarItem(
                            selected = cur.tab == s.tab, onClick = { Nav.tab(s) },
                            icon = { Text(icon, fontSize = 18.sp) }, label = { Text(label, fontSize = 11.sp) },
                            colors = NavigationBarItemDefaults.colors(selectedTextColor = K.accent, selectedIconColor = K.accent, indicatorColor = K.accentSoft),
                        )
                    }
                }
            },
        ) { pad ->
            Box(Modifier.padding(pad).fillMaxSize().background(K.paper)) {
                key(cur) {
                    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 14.dp, vertical = 8.dp)) {
                        Repo.loadError?.let { Banner(it, warn = true) }
                        when (cur) {
                            Screen.Today -> TodayScreen()
                            is Screen.DayPage -> DayScreen(cur.date, cur.plan)
                            Screen.History -> HistoryScreen()
                            Screen.StarList -> StarListScreen()
                            is Screen.StarRun -> StarRunScreen(cur.id)
                            Screen.Experiences -> ExperiencesScreen()
                            Screen.Balance -> BalanceScreen()
                            Screen.Settings -> SettingsScreen()
                        }
                        Spacer(Modifier.height(24.dp))
                    }
                }
            }
        }
    }
}

fun today() = Dates.today()
