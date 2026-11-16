plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
    id("org.jetbrains.kotlin.plugin.serialization")
}

android {
    namespace = "app.kimiagar"
    compileSdk = 35

    defaultConfig {
        applicationId = "app.kimiagar"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0.0"
    }

    // Star Rewriting texts: single source of truth shared with the desktop app.
    sourceSets["main"].assets.srcDirs("../../src/config")

    // One fixed key for every build (any PC, debug or release), so a new version always installs
    // over the old one. Android refuses updates signed with a different key, and uninstalling would delete the data.
    signingConfigs {
        create("kimiagar") {
            storeFile = rootProject.file("kimiagar.keystore")
            storePassword = "kimiagar-local"
            keyAlias = "kimiagar"
            keyPassword = "kimiagar-local"
        }
    }
    buildTypes {
        debug { signingConfig = signingConfigs.getByName("kimiagar") }
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("kimiagar")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    buildFeatures { compose = true }
    testOptions {
        unitTests.isIncludeAndroidResources = true
        unitTests.all { it.systemProperty("robolectric.dependency.repo.url", "https://maven-central.storage-download.googleapis.com/maven2") }
    }
}

dependencies {
    val bom = platform("androidx.compose:compose-bom:2024.12.01")
    implementation(bom)
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.3")
    testImplementation("junit:junit:4.13.2")
    // UI flow tests run on the JVM with Robolectric (no emulator needed)
    testImplementation(bom)
    testImplementation("androidx.compose.ui:ui-test-junit4")
    testImplementation("org.robolectric:robolectric:4.14.1")
    debugImplementation("androidx.compose.ui:ui-test-manifest")
}
