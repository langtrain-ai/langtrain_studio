#!/usr/bin/env node
/**
 * One version for all of Langtrain Studio: the app on macOS, Windows and Linux,
 * and the engine inside it. VERSION at the repo root is the source of truth;
 * this script writes it everywhere else that carries a version.
 *
 *   node scripts/version.mjs              # print the version
 *   node scripts/version.mjs set 1.1.0    # change it everywhere
 *   node scripts/version.mjs sync         # re-apply VERSION everywhere
 *   node scripts/version.mjs check        # exit 1 if anything disagrees (CI)
 */
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const read = (p) => readFileSync(join(root, p), "utf8");
const write = (p, s) => writeFileSync(join(root, p), s);

const APP_CRATE = "langtrain-studio-desktop";
const ENGINE_CRATES = ["langtrain", "langtrain-engine", "langtrain-native", "langtrain-server", "langtrain-web"];

/** Every place a version lives: how to read it and how to write it. */
const json = (file) => ({
  file,
  get: () => JSON.parse(read(file)).version,
  set: (v) => {
    const text = read(file);
    // Replace in place to keep the file's formatting.
    write(file, text.replace(/("version"\s*:\s*")[^"]*(")/, `$1${v}$2`));
  },
});

/** `version = "..."` as the first version key after `[section]`. */
const toml = (file, section) => {
  const re = new RegExp(`(^\\[${section.replace(/[.[\]]/g, "\\$&")}\\][^\\[]*?^version\\s*=\\s*")([^"]*)(")`, "ms");
  return {
    file: `${file} [${section}]`,
    get: () => read(file).match(re)?.[2],
    set: (v) => write(file, read(file).replace(re, `$1${v}$3`)),
  };
};

/** Lockfile entries for crates in this repo, so `cargo --locked` keeps passing. */
const lock = (file, crates) => ({
  file: `${file} (${crates.join(", ")})`,
  get: () => {
    const found = crates.map((c) => read(file).match(new RegExp(`name = "${c}"\\nversion = "([^"]*)"`))?.[1]);
    return found.every((v) => v === found[0]) ? found[0] : `mixed (${found.join(", ")})`;
  },
  set: (v) => {
    let text = read(file);
    for (const c of crates) text = text.replace(new RegExp(`(name = "${c}"\\nversion = ")[^"]*(")`), `$1${v}$2`);
    write(file, text);
  },
});

const TARGETS = [
  json("app/package.json"),
  json("app/src-tauri/tauri.conf.json"),
  toml("app/src-tauri/Cargo.toml", "package"),
  lock("app/src-tauri/Cargo.lock", [APP_CRATE]),
  toml("engine/Cargo.toml", "workspace.package"),
  lock("engine/Cargo.lock", ENGINE_CRATES),
];

const current = () => read("VERSION").trim();
const valid = (v) => /^\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?$/.test(v);

function sync(v) {
  for (const t of TARGETS) t.set(v);
  const wrong = TARGETS.filter((t) => t.get() !== v);
  if (wrong.length) {
    console.error(`Could not update: ${wrong.map((t) => t.file).join(", ")}`);
    process.exit(1);
  }
}

const [cmd, arg] = process.argv.slice(2);
switch (cmd) {
  case undefined:
    console.log(current());
    break;
  case "set":
    if (!valid(arg ?? "")) {
      console.error("Usage: node scripts/version.mjs set <major.minor.patch>");
      process.exit(1);
    }
    write("VERSION", `${arg}\n`);
    sync(arg);
    console.log(`Langtrain Studio is now ${arg} everywhere. Tag the release with: git tag v${arg}`);
    break;
  case "sync":
    sync(current());
    console.log(`Applied ${current()} everywhere.`);
    break;
  case "check": {
    const v = current();
    const wrong = TARGETS.filter((t) => t.get() !== v);
    for (const t of wrong) console.error(`${t.file}: ${t.get()} (VERSION says ${v})`);
    if (wrong.length) {
      console.error("Run: node scripts/version.mjs sync");
      process.exit(1);
    }
    const tag = process.env.GITHUB_REF_TYPE === "tag" ? process.env.GITHUB_REF_NAME : null;
    if (tag && tag !== `v${v}`) {
      console.error(`Tag ${tag} does not match VERSION ${v}.`);
      process.exit(1);
    }
    console.log(`All versions are ${v}.`);
    break;
  }
  default:
    console.error(`Unknown command: ${cmd}`);
    process.exit(1);
}
