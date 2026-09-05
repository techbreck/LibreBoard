plugins {
    id("com.android.application")
}

val maximumArchiveBytes = 28L * 1024L * 1024L
val archiveProperty = providers.gradleProperty("libreboardContextModelArchive").orNull
    ?: error("-PlibreboardContextModelArchive must name the signed context-en-de .lbmodel archive")
val contextModelArchive = rootProject.file(archiveProperty)
require(contextModelArchive.isFile) { "Context model archive does not exist: $contextModelArchive" }
require(contextModelArchive.extension == "lbmodel") { "Context model archive must end in .lbmodel" }
require(contextModelArchive.length() in 1..maximumArchiveBytes) {
    "Context model archive must be non-empty and at most $maximumArchiveBytes bytes"
}

val generatedAssets = layout.buildDirectory.dir("generated/modelpack/assets")
val prepareContextModelArchive by tasks.registering(Copy::class) {
    from(contextModelArchive)
    into(generatedAssets)
    rename { "model.lbmodel" }
}

android {
    namespace = "org.libreboard.model.en_de"
    compileSdk = 36

    defaultConfig {
        applicationId = "org.libreboard.model.en_de"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "1.0.0"
    }

    buildFeatures {
        buildConfig = false
    }

    sourceSets.getByName("main").assets.srcDir(generatedAssets)
    androidResources {
        // AssetManager.openFd requires the model archive to be stored rather than deflated.
        noCompress += "lbmodel"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    testOptions {
        unitTests.isIncludeAndroidResources = true
    }

    dependenciesInfo {
        includeInApk = false
        includeInBundle = false
    }

    lint {
        abortOnError = true
    }
}

tasks.named("preBuild").configure {
    dependsOn(prepareContextModelArchive)
}

dependencies {
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.robolectric:robolectric:4.16.1")
}
