# Langtrain Studio

One desktop app for **macOS, Windows and Linux**, with the Langtrain engine
built in. Every change here ships to all three platforms from the same code.

```
Langtrain_Studio/
├── VERSION        one version for everything (see "Releasing")
├── app/           the desktop app: Tauri 2 + React. Same code on every OS
├── engine/        the Langtrain engine (Rust): OpenAI/Ollama-compatible
│                  server, native inference, CPU fine-tuning (LangTune)
├── scripts/
│   ├── version.mjs       set / check the version everywhere
│   └── build-engine.sh   build the engine into the app as its sidecar
└── .github/workflows/
    ├── ci.yml       every PR: version check, app, engine on 4 OSes, LangTune
    └── release.yml  a v* tag: app installers for all platforms + engine CLI
```

## How the pieces fit

The app bundles the engine as a sidecar (`langtrain serve` on
`127.0.0.1:11435`). Local models in the app run on it. Both use
`~/.langtrain/models`, so a model downloaded in the app is immediately
available to the engine, and to `langtrain` on the command line.

## Develop

```bash
scripts/build-engine.sh            # once, and after engine changes (adds app/src-tauri/binaries/)
cd app && pnpm install
pnpm tauri dev                     # the app, with live reload
```

`scripts/build-engine.sh --fast` skips llama.cpp for a quicker build; the
native engine still runs most models.

Engine on its own: see `engine/README.md` (`cargo run -p langtrain -- serve`).

## Releasing

```bash
node scripts/version.mjs set 1.1.0     # updates the app, Tauri, Cargo and lockfiles
git commit -am "Release 1.1.0"
git tag v1.1.0 && git push origin main v1.1.0
```

The tag builds and publishes, in one GitHub release:

| Platform | Files |
|---|---|
| macOS (Apple silicon, Intel) | `.dmg` |
| Windows | `.exe`, `.msi` |
| Linux | `.AppImage`, `.deb` |
| Engine CLI | `langtrain-v1.1.0-<target>.tar.gz` / `.zip` for 6 targets |

CI refuses a tag that does not match `VERSION`.

macOS builds are signed and notarised when the `APPLE_*` secrets are set;
without them they are still built, and macOS asks for right-click > Open on
first launch. `RELEASE_API_KEY` registers the downloads with api.langtrain.xyz.

## History

This repo combines two earlier ones, with their full history:

- `app/` was **langtrain_studio_desktop** (Windows and Linux only; macOS was
  switched off in favour of the Swift app).
- `engine/` was **langtrain-studio**. Its separate desktop shell
  (`engine/studio/`) was removed: `app/` is now the one desktop app.

The native SwiftUI macOS app (**langtrain_studio**) is archived, not merged:
it is proprietary and stays in its own private repository.
