#!/usr/bin/env bash
# Compiles and runs CdkParityReference.java against a CDK fat jar and (re)writes
# crates/ferrum_core/tests/data/cdk_parity_reference.csv.
#
#     scripts/cdk_reference/run.sh
#     CDK_JAR=/path/to/cdk-2.x.jar scripts/cdk_reference/run.sh
#
# Requires Java >= 11 (javac, java). Class files go to scripts/cdk_reference/build/ (gitignored).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
CDK_JAR="${CDK_JAR:-$HOME/dev_new/cdk/bundle/target/cdk-2.13-SNAPSHOT.jar}"
BUILD="$HERE/build"

if [[ ! -f "$CDK_JAR" ]]; then
    echo "CDK jar not found: $CDK_JAR (set CDK_JAR)" >&2
    exit 1
fi

mkdir -p "$BUILD"
javac -nowarn -encoding UTF-8 -cp "$CDK_JAR" -d "$BUILD" "$HERE/CdkParityReference.java"
java -Xss64m -Xmx2g -cp "$BUILD:$CDK_JAR" CdkParityReference "$ROOT"
