import com.android.build.api.variant.ApplicationVariant
import groovy.json.JsonSlurper
import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import java.security.MessageDigest

val pinnedOnnxRuntimeCommit = "8c546c37b43caaca1fa25db430dab94b901cf277"
val pinnedNdkRevision = "28.0.13004108"
val releaseBaseVersionCode = 1
val supportedApplicationAbis = listOf("armeabi-v7a", "arm64-v8a", "x86", "x86_64")
val applicationAbiVersionOffsets = mapOf(
    "armeabi-v7a" to 1,
    "arm64-v8a" to 2,
    "x86" to 3,
    "x86_64" to 4,
)
val targetApplicationAbi = providers.gradleProperty("libreboardTargetAbi").orNull?.also { abi ->
    require(abi in supportedApplicationAbis) {
        "libreboardTargetAbi must be one of: ${supportedApplicationAbis.joinToString()}"
    }
}
val packagedVersionCode = releaseBaseVersionCode * 100 + (
    targetApplicationAbi?.let(applicationAbiVersionOffsets::getValue) ?: 99
)

private fun sha256(file: File): String {
    val digest = MessageDigest.getInstance("SHA-256")
    file.inputStream().buffered().use { input ->
        val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
        while (true) {
            val count = input.read(buffer)
            if (count < 0) break
            digest.update(buffer, 0, count)
        }
    }
    return digest.digest().joinToString("") { "%02x".format(it) }
}

val onnxRuntimeAarProperty = providers.gradleProperty("libreboardOnnxRuntimeAar").orNull
val onnxRuntimeManifestProperty = providers.gradleProperty("libreboardOnnxRuntimeManifest").orNull
val onnxRuntimeOperatorsProperty = providers.gradleProperty("libreboardOnnxRuntimeOperators").orNull
val onnxRuntimeProperties = listOf(
    onnxRuntimeAarProperty,
    onnxRuntimeManifestProperty,
    onnxRuntimeOperatorsProperty,
)
require(onnxRuntimeProperties.all { it == null } || onnxRuntimeProperties.all { it != null }) {
    "The source-built ONNX Runtime AAR, build manifest, and operator config must be supplied together"
}
val verifiedOnnxRuntimeAar = onnxRuntimeAarProperty?.let { aarPath ->
    val aar = rootProject.file(aarPath)
    val manifestFile = rootProject.file(requireNotNull(onnxRuntimeManifestProperty))
    val operatorsFile = rootProject.file(requireNotNull(onnxRuntimeOperatorsProperty))
    val runtimeSettings = rootProject.file("runtime/onnxruntime/build-settings.json")
    val runtimeBuilder = rootProject.file("tools/build_onnxruntime_android.py")
    require(aar.isFile && manifestFile.isFile && operatorsFile.isFile) {
        "The source-built ONNX Runtime artifacts are missing"
    }
    require(runtimeSettings.isFile) { "The audited ONNX Runtime build settings are missing" }
    require(runtimeBuilder.isFile) { "The audited ONNX Runtime builder is missing" }
    @Suppress("UNCHECKED_CAST")
    val manifest = JsonSlurper().parse(manifestFile) as? Map<String, Any?>
        ?: error("The ONNX Runtime build manifest is invalid")
    require(manifest.keys == setOf(
        "schemaVersion",
        "sourceCommit",
        "ndkRevision",
        "pythonVersion",
        "pythonPackages",
        "sourceDateEpoch",
        "buildToolSha256",
        "settingsSha256",
        "operatorsSha256",
        "aarSha256",
        "nativeLibraries",
    )) { "The ONNX Runtime build manifest has unexpected fields" }
    require(manifest["schemaVersion"] == 2) { "Unsupported ONNX Runtime build manifest" }
    require(manifest["sourceCommit"] == pinnedOnnxRuntimeCommit) {
        "The ONNX Runtime AAR was built from an unapproved source commit"
    }
    require(manifest["ndkRevision"] == pinnedNdkRevision) {
        "The ONNX Runtime AAR was built with a different NDK"
    }
    require(manifest["pythonVersion"] == "3.11") {
        "The ONNX Runtime AAR was built with an unapproved Python interpreter"
    }
    require(manifest["pythonPackages"] == mapOf(
        "flatbuffers" to "25.12.19",
        "numpy" to "2.2.6",
        "onnxruntime" to "1.26.0",
        "packaging" to "26.3",
        "protobuf" to "7.36.1",
    )) { "The ONNX Runtime AAR was built with unapproved Python packages" }
    require((manifest["sourceDateEpoch"] as? Number)?.toLong()?.let { it > 0 } == true) {
        "The ONNX Runtime AAR has an invalid reproducibility epoch"
    }
    require(manifest["buildToolSha256"] == sha256(runtimeBuilder)) {
        "The ONNX Runtime AAR was built with a different builder"
    }
    require(manifest["settingsSha256"] == sha256(runtimeSettings)) {
        "The ONNX Runtime AAR was built with different settings"
    }
    require(manifest["operatorsSha256"] == sha256(operatorsFile)) {
        "The ONNX Runtime AAR was reduced for a different operator set"
    }
    require(manifest["aarSha256"] == sha256(aar)) {
        "The ONNX Runtime AAR hash does not match its build manifest"
    }
    aar
}

val swipeModelArchiveProperty = providers.gradleProperty("libreboardSwipeModelArchive").orNull
val modelPublicKeyProperty = providers.gradleProperty("libreboardModelPublicKey").orNull
require((swipeModelArchiveProperty == null) == (modelPublicKeyProperty == null)) {
    "The signed swipe model archive and project public key must be supplied together"
}
require(swipeModelArchiveProperty == null || verifiedOnnxRuntimeAar != null) {
    "A signed swipe model requires the source-built ONNX Runtime artifact set"
}
val verifiedSwipeModelArchive = swipeModelArchiveProperty?.let { path ->
    rootProject.file(path).also { archive ->
        require(archive.isFile && archive.extension == "lbmodel") {
            "The signed swipe model archive is missing or has the wrong extension"
        }
        require(archive.length() in 1..(3L * 1024L * 1024L)) {
            "The signed swipe model archive must be non-empty and at most 3 MiB"
        }
    }
}
val verifiedModelPublicKey = modelPublicKeyProperty?.let { path ->
    rootProject.file(path).also { key ->
        require(key.isFile && key.length() in 1..(8L * 1024L)) {
            "The model-signing public key is missing, empty, or oversized"
        }
    }
}
val generatedSignedModelAssets = layout.buildDirectory.dir("generated/libreboard/signedModelAssets")
val cleanSignedModelAssets by tasks.registering(Delete::class) {
    // Always remove generated assets first. A core-only build must not reuse files produced by a
    // previous model-qualified invocation under a different Gradle configuration-cache key.
    delete(generatedSignedModelAssets)
}
val prepareSignedModelAssets by tasks.registering(Sync::class) {
    dependsOn(cleanSignedModelAssets)
    into(generatedSignedModelAssets)
    verifiedSwipeModelArchive?.let { archive ->
        from(archive) {
            into("models")
            rename { "swipe-latin-v1.lbmodel" }
        }
    }
    verifiedModelPublicKey?.let { key ->
        from(key) {
            into("models")
            rename { "libreboard-model-signing-public.der" }
        }
    }
}

plugins {
    id("com.android.application")
    kotlin("android")
    kotlin("plugin.serialization") version "2.3.20"
    kotlin("plugin.compose") version "2.3.20"
}

android {
    compileSdk = 36
    // Default stays debugNoMinify for the connected suite. GrapheneOS representative-latency
    // runs need a non-debuggable testOnly APK; pass -PlibreboardTestBuildType=measure so the
    // androidTest APK is built against that variant (run-as is unavailable there).
    testBuildType = providers.gradleProperty("libreboardTestBuildType").orElse("debugNoMinify").get()

    defaultConfig {
        applicationId = "org.libreboard.keyboard"
        minSdk = 26
        targetSdk = 36
        // Keep this unencoded base visible to F-Droid's update checker. Variant outputs use
        // 100 * base + 1..4 for one-ABI builds and +99 for the universal GitHub artifact.
        versionCode = releaseBaseVersionCode
        versionName = "0.1.0-alpha01"
        buildConfigField("boolean", "LIBREBOARD_ONNX_RUNTIME_PACKAGED", (verifiedOnnxRuntimeAar != null).toString())
        buildConfigField("boolean", "LIBREBOARD_SIGNED_MODELS_PACKAGED", (verifiedSwipeModelArchive != null).toString())
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        ndk {
            abiFilters.clear()
            abiFilters.addAll(targetApplicationAbi?.let(::listOf) ?: supportedApplicationAbis)
        }
        proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = false
            isDebuggable = false
            isJniDebuggable = false
        }
        debug {
            // "normal" debug has minify for smaller APK to fit the GitHub 25 MB limit when zipped
            // and for better performance in case users want to install a debug APK
            isMinifyEnabled = true
            isJniDebuggable = false
            applicationIdSuffix = ".debug"
        }
        create("runTests") { // build variant for running tests on CI that skips tests known to fail
            isMinifyEnabled = false
            isJniDebuggable = false
        }
        create("debugNoMinify") { // for faster builds in IDE
            isDebuggable = true
            isMinifyEnabled = false
            isJniDebuggable = false
            signingConfig = signingConfigs.getByName("debug")
            applicationIdSuffix = ".debug"
        }
        // Non-debuggable testOnly sibling of debugNoMinify. GrapheneOS disables the JIT and
        // caps debuggable packages to verify-only AOT, so instrumented debug replays run
        // interpreted. This variant is adb-installable (`adb install -t`) and AOT-compiles.
        create("measure") {
            initWith(getByName("debugNoMinify"))
            isDebuggable = false
            isJniDebuggable = false
        }

        androidComponents.onVariants { variant: ApplicationVariant ->
            if (variant.buildType == "debug") {
                // got a little too big for GitHub after some dependency upgrades, so we remove the largest dictionary
                variant.androidResources.ignoreAssetsPatterns = listOf("main_ro.dict")
                variant.proguardFiles = emptyList()
                //noinspection ProguardAndroidTxtUsage we intentionally use the "normal" file here
                variant.proguardFiles.add(project.layout.buildDirectory.file(project.buildFile.parent + "/dontoptimize.pro"))
                variant.proguardFiles.add(project.layout.buildDirectory.file(project.buildFile.parent + "/proguard-rules.pro"))
            }
            variant.outputs.forEach { output ->
                output.versionCode.set(packagedVersionCode)
                if (output is com.android.build.api.variant.impl.VariantOutputImpl) {
                    val abiSuffix = targetApplicationAbi?.let { "-$it" }.orEmpty()
                    output.outputFileName =
                        "LibreBoard_${defaultConfig.versionName}-${variant.buildType}$abiSuffix.apk"
                }
            }
        }
    }

    buildFeatures {
        viewBinding = true
        buildConfig = true
        compose = true
    }

    externalNativeBuild {
        ndkBuild {
            path = File("src/main/jni/Android.mk")
        }
    }
    ndkVersion = pinnedNdkRevision

    packaging {
        jniLibs {
            // shrinks APK by 3 MB, zipped size unchanged
            useLegacyPackaging = true
        }
    }

    sourceSets.getByName("main").assets.srcDir(generatedSignedModelAssets)

    testOptions {
        unitTests {
            isIncludeAndroidResources = true
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlin {
        target {
            compilerOptions {
                jvmTarget.set(JvmTarget.JVM_17)
            }
        }
    }

    // see https://github.com/HeliBorg/HeliBoard/issues/477
    dependenciesInfo {
        includeInApk = false
        includeInBundle = false
    }

    sourceSets {
        getByName("measure") {
            java.srcDir("src/debugNoMinify/java")
            res.srcDir("src/debugNoMinify/res")
        }
    }

    namespace = "helium314.keyboard.latin"
    lint {
        abortOnError = true
    }
}

tasks.named("preBuild").configure {
    dependsOn(prepareSignedModelAssets)
}

dependencies {
    verifiedOnnxRuntimeAar?.let { implementation(files(it)) }

    // androidx
    implementation("androidx.core:core-ktx:1.17.0") // 1.18.0 requires minSdk 23
    implementation("androidx.recyclerview:recyclerview:1.4.0")
    implementation("androidx.autofill:autofill:1.3.0")
    implementation("androidx.viewpager2:viewpager2:1.1.0")

    // kotlin
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.11.0")

    // compose
    coreLibraryDesugaring("com.android.tools:desugar_jdk_libs:2.1.5")
    // newer than 2025.11.01 contains androidx.compose.material:material-android:1.10.0, which requires minSdk 23
    // maybe it's possible to use tools:overrideLibrary="androidx.compose.material" as it's not used explicitly, but probably this is just going to crash
    implementation(platform("androidx.compose:compose-bom:2025.11.01"))
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui-tooling-preview")
    debugImplementation("androidx.compose.ui:ui-tooling")
    "debugNoMinifyImplementation"("androidx.compose.ui:ui-tooling")
    implementation("androidx.navigation:navigation-compose:2.9.8")
    implementation("sh.calvin.reorderable:reorderable:3.1.0") // for easier re-ordering
    implementation("com.github.skydoves:colorpicker-compose:1.1.3") // for user-defined colors

    // test
    testImplementation(kotlin("test"))
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.mockito:mockito-core:5.23.0")
    testImplementation("org.robolectric:robolectric:4.16.1")
    testImplementation("androidx.test:runner:1.7.0")
    testImplementation("androidx.test:core:1.7.0")
    androidTestImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test:runner:1.7.0")
    androidTestImplementation("androidx.test:core:1.7.0")
}
