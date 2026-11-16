package app.kimiagar

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import app.kimiagar.data.Repo
import app.kimiagar.ui.KimiagarApp

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Repo.init(applicationContext)
        setContent { KimiagarApp() }
    }

    // Always write pending changes when the app leaves the screen.
    override fun onStop() {
        super.onStop()
        Repo.flush()
    }
}
