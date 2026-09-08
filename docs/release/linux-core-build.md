# Linux core build environment

`runtime/android/Dockerfile.core` prepares an x86-64 Linux JDK 17, Android API 36, Build Tools
35.0.0 and NDK 28.0.13004108 environment. It pins the base image digest and Command-line Tools
12.0 archive SHA-256. The latter was downloaded from Google and matched against its repository
metadata's 153,607,504-byte size and SHA-1 before its SHA-256 was recorded. Gradle independently
verifies its own dependencies, including the Linux AAPT2 artifact.

Copy licenses from an SDK installation whose terms the operator has already accepted; the recipe
does not automatically accept license terms:

```sh
mkdir -p build/linux-core-context
cp -R "$ANDROID_HOME/licenses" build/linux-core-context/licenses
docker build --platform linux/amd64 -f runtime/android/Dockerfile.core \
  -t libreboard-core-release:local build/linux-core-context
docker image inspect --format '{{.Id}}' libreboard-core-release:local
```

Build a clean source archive with an explicitly recorded commit. Give each reproduction its own
archive directory, and retain the logs and output APKs. A shared Gradle download cache is allowed;
build and configuration caches remain disabled. Reuse the cache volume sequentially. Concurrent
containers need separate cache volumes because Gradle's lock-owner coordination cannot cross their
PID/network namespaces; sharing a live cache can time out before configuration:

```sh
mkdir -p build/linux-core-source
git rev-parse HEAD > build/linux-core-source-commit.txt
git archive HEAD | tar -x -C build/linux-core-source
docker run --rm --platform linux/amd64 --cpus 2 --memory 3g \
  --mount "type=bind,src=$PWD/build/linux-core-source,dst=/source" \
  --mount type=volume,src=libreboard-core-gradle-cache,dst=/root/.gradle \
  libreboard-core-release:local ./gradlew :app:assembleRelease --no-daemon \
  --max-workers=1 --no-build-cache --no-configuration-cache --dependency-verification strict
```

Record the resolved image ID and JDK/SDK revisions with every comparison. Debian package repositories
and SDK platform revisions can move, so this is not a claim that the environment image itself is
byte-reproducible across dates. On ARM hosts the x86 toolchain runs through emulation and the first
Gradle dependency transforms can take several minutes. The pinned NDK does not directly support an
ARM Linux host.

The output is an unsigned core-only APK. Verify it with `tools/verify_release.py --apk`, then compare
the bytes with the second independent build. This recipe does not sign APKs, package models, or
satisfy the model-quality and physical-device gates.
