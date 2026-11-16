package app.kimiagar.ui

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalLayoutDirection
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.LayoutDirection
import androidx.compose.ui.unit.sp
import app.kimiagar.R

object K {
    val paper = Color(0xFFF7F5F0)
    val card = Color.White
    val ink = Color(0xFF22232B)
    val muted = Color(0xFF6D6E78)
    val line = Color(0xFFD9D6CF)
    val accent = Color(0xFF27357A)
    val accentSoft = Color(0xFFE7E9F5)
    val good = Color(0xFF2F7A4F)
    val goodSoft = Color(0xFFE3F1E8)
    val bad = Color(0xFFB3372D)
    val warnSoft = Color(0xFFFBF1D9)
    val red = Color(0xFFC4122F)
}

val Vazirmatn = FontFamily(Font(R.font.vazirmatn_regular, FontWeight.Normal), Font(R.font.vazirmatn_bold, FontWeight.Bold))

@Composable
fun KimiagarTheme(content: @Composable () -> Unit) {
    val base = TextStyle(fontFamily = Vazirmatn, color = K.ink)
    val typo = Typography(
        bodyLarge = base.copy(fontSize = 15.sp, lineHeight = 24.sp),
        bodyMedium = base.copy(fontSize = 14.sp, lineHeight = 22.sp),
        bodySmall = base.copy(fontSize = 12.sp, lineHeight = 18.sp),
        titleLarge = base.copy(fontSize = 19.sp, fontWeight = FontWeight.Bold),
        titleMedium = base.copy(fontSize = 16.sp, fontWeight = FontWeight.Bold),
        titleSmall = base.copy(fontSize = 14.sp, fontWeight = FontWeight.Bold),
        labelLarge = base.copy(fontSize = 14.sp, fontWeight = FontWeight.Bold),
        labelMedium = base.copy(fontSize = 12.sp),
        labelSmall = base.copy(fontSize = 11.sp),
    )
    MaterialTheme(
        colorScheme = lightColorScheme(primary = K.accent, onPrimary = Color.White, background = K.paper, surface = K.card, onSurface = K.ink, secondaryContainer = K.accentSoft),
        typography = typo,
    ) {
        CompositionLocalProvider(LocalLayoutDirection provides LayoutDirection.Rtl, content = content)
    }
}
